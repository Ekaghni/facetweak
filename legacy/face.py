import cv2
import numpy as np
import dlib
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class NormalizedLandmark:
    """Represents a 3D normalized landmark point"""
    x: float
    y: float
    z: float


@dataclass
class FaceCoordinates:
    """Face bounding box coordinates"""
    left: int
    top: int
    right: int
    bottom: int


class AppliedEffect:
    """Represents an applied effect with its progress"""
    def __init__(self, effect_func: Callable, progress: int):
        self.effect = effect_func
        self.progress = progress
    
    def __eq__(self, other):
        if isinstance(other, AppliedEffect):
            return self.effect == other.effect
        elif callable(other):
            return self.effect == other
        return False


class Effect:
    """Effect definition with callback function"""
    def __init__(self, name: str, callback_function: Callable):
        self.name = name
        self.callback_function = callback_function


class ImageEffects:
    """Advanced image effects with natural-looking transformations"""
    
    @staticmethod
    def gaussian_falloff(distance: np.ndarray, radius: float, smoothness: float = 0.3) -> np.ndarray:
        """Create smooth Gaussian-like falloff for natural blending"""
        normalized_distance = distance / radius
        # Use sigmoid-like function for smooth falloff
        falloff = np.exp(-0.5 * (normalized_distance / smoothness) ** 2)
        return np.clip(falloff, 0, 1)
    
    @staticmethod
    def natural_magnify(image: np.ndarray, center_x: int, center_y: int, radius: int, strength: float) -> np.ndarray:
        """Natural-looking magnification with smooth edges"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Create smooth falloff mask
        falloff = ImageEffects.gaussian_falloff(distance, radius, 0.4)
        
        # Apply magnification with smooth transition
        # Use a more natural magnification formula
        scale_factor = 1.0 + strength * falloff * (1.0 - distance / (radius * 2))
        scale_factor = np.maximum(scale_factor, 0.5)  # Prevent extreme shrinking
        
        # Calculate new coordinates
        x_new = center_x + dx / scale_factor
        y_new = center_y + dy / scale_factor
        
        # Ensure coordinates are within bounds and proper type
        x_new = np.clip(x_new, 0, width - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, height - 1).astype(np.float32)
        
        # Apply the transformation
        transformed = cv2.remap(image, x_new, y_new, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with original using smooth alpha
        alpha = falloff[:, :, np.newaxis]
        result = (transformed * alpha + image * (1 - alpha)).astype(np.uint8)
        
        return result
    
    @staticmethod
    def natural_shift(image: np.ndarray, center_x: int, center_y: int, radius: int, shift_x: float, shift_y: float) -> np.ndarray:
        """Natural-looking region shifting with smooth falloff"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Create smooth falloff
        falloff = ImageEffects.gaussian_falloff(distance, radius, 0.5)
        
        # Apply gradual shift
        x_new = x - shift_x * falloff
        y_new = y - shift_y * falloff
        
        # Ensure coordinates are within bounds and proper type
        x_new = np.clip(x_new, 0, width - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, height - 1).astype(np.float32)
        
        # Apply the transformation with cubic interpolation for smoothness
        result = cv2.remap(image, x_new, y_new, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def elliptical_magnify(image: np.ndarray, center_x: int, center_y: int, 
                          width_radius: int, height_radius: int, strength: float, angle: float = 0) -> np.ndarray:
        """Elliptical magnification for more natural lip/eye shapes"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Calculate rotated elliptical distance
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        
        # Rotate coordinates
        dx = x - center_x
        dy = y - center_y
        dx_rot = dx * cos_angle + dy * sin_angle
        dy_rot = -dx * sin_angle + dy * cos_angle
        
        # Elliptical distance
        distance = np.sqrt((dx_rot / width_radius) ** 2 + (dy_rot / height_radius) ** 2)
        
        # Create smooth falloff
        falloff = np.exp(-0.5 * (distance / 0.5) ** 2)
        falloff = np.clip(falloff, 0, 1)
        
        # Apply magnification
        scale_factor = 1.0 + strength * falloff * (1.0 - distance / 2)
        scale_factor = np.maximum(scale_factor, 0.5)
        
        # Calculate new coordinates
        x_new = center_x + dx / scale_factor
        y_new = center_y + dy / scale_factor
        
        # Ensure coordinates are within bounds and proper type
        x_new = np.clip(x_new, 0, width - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, height - 1).astype(np.float32)
        
        # Apply the transformation
        transformed = cv2.remap(image, x_new, y_new, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with original
        alpha = falloff[:, :, np.newaxis]
        result = (transformed * alpha + image * (1 - alpha)).astype(np.uint8)
        
        return result
    
    @staticmethod
    def directional_stretch(image: np.ndarray, center_x: int, center_y: int, 
                           radius: int, stretch_x: float, stretch_y: float) -> np.ndarray:
        """Stretch effect in specific directions"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Create smooth falloff
        falloff = ImageEffects.gaussian_falloff(distance, radius, 0.4)
        
        # Apply directional stretching
        # Positive values stretch outward, negative values compress
        x_scale = 1.0 - stretch_x * falloff
        y_scale = 1.0 - stretch_y * falloff
        
        # Calculate new coordinates
        x_new = center_x + dx * x_scale
        y_new = center_y + dy * y_scale
        
        # Ensure coordinates are within bounds and proper type
        x_new = np.clip(x_new, 0, width - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, height - 1).astype(np.float32)
        
        # Apply the transformation
        result = cv2.remap(image, x_new, y_new, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def rotate_region(image: np.ndarray, center_x: int, center_y: int, 
                     radius: int, angle_degrees: float) -> np.ndarray:
        """Rotate a region around its center"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Create smooth falloff
        falloff = ImageEffects.gaussian_falloff(distance, radius, 0.5)
        
        # Apply rotation with falloff
        angle_rad = np.radians(angle_degrees * falloff)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Rotate coordinates
        x_new = center_x + dx * cos_angle - dy * sin_angle
        y_new = center_y + dx * sin_angle + dy * cos_angle
        
        # Ensure coordinates are within bounds and proper type
        x_new = np.clip(x_new, 0, width - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, height - 1).astype(np.float32)
        
        # Apply the transformation
        result = cv2.remap(image, x_new, y_new, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def multi_point_magnify(image: np.ndarray, points: List[Tuple[int, int]], radius: int, strength: float) -> np.ndarray:
        """Magnify multiple points with blended effects"""
        result = image.copy().astype(np.float32)
        height, width = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        
        total_weight = np.zeros((height, width), dtype=np.float32)
        weighted_x = np.zeros((height, width), dtype=np.float32)
        weighted_y = np.zeros((height, width), dtype=np.float32)
        
        for center_x, center_y in points:
            # Calculate distance from this center
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Create weight map for this point
            weight = ImageEffects.gaussian_falloff(distance, radius, 0.4)
            
            # Calculate transformation for this point
            scale_factor = 1.0 + strength * weight * (1.0 - distance / (radius * 2))
            scale_factor = np.maximum(scale_factor, 0.5)
            
            # Calculate new coordinates for this transformation
            local_x_new = center_x + dx / scale_factor
            local_y_new = center_y + dy / scale_factor
            
            # Accumulate weighted transformations
            total_weight += weight
            weighted_x += local_x_new * weight
            weighted_y += local_y_new * weight
        
        # Avoid division by zero
        total_weight = np.maximum(total_weight, 1e-6)
        
        # Calculate final coordinates
        final_x = np.where(total_weight > 0.01, weighted_x / total_weight, x)
        final_y = np.where(total_weight > 0.01, weighted_y / total_weight, y)
        
        # Ensure coordinates are within bounds and proper type
        final_x = np.clip(final_x, 0, width - 1).astype(np.float32)
        final_y = np.clip(final_y, 0, height - 1).astype(np.float32)
        
        # Apply the transformation
        transformed = cv2.remap(image, final_x, final_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with original using accumulated weights
        alpha = np.clip(total_weight, 0, 1)[:, :, np.newaxis]
        result = (transformed * alpha + image * (1 - alpha)).astype(np.uint8)
        
        return result


class Landmarks:
    """Main class representing face landmarks and applied effects"""
    
    def __init__(self, angle: float, face_coordinates: FaceCoordinates, 
                 normalized_landmarks: List[NormalizedLandmark]):
        self.angle = angle
        self.face_coordinates = face_coordinates
        self.normalized_landmarks = normalized_landmarks
        self.face_clipped = None
        self.applied_effects = []
        self.redo_stack = []
        self.face_scale_ratio = 1.0
    
    def set_clipped_rotated_face(self, main_image: np.ndarray):
        """Extract and prepare the face region for processing"""
        if self.face_clipped is not None:
            return
        
        # Just store reference to original coordinates - no clipping for now
        self.face_clipped = True  # Mark as processed
    
    def apply_effect(self, effect_func: Optional[Callable], progress: int) -> np.ndarray:
        """Apply effect to the full image (simplified approach)"""
        # For now, return a placeholder - effects will be applied directly to full image
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def get_current_effect_value(self, effect_func: Callable) -> int:
        """Get current progress value for an effect"""
        for applied_effect in self.applied_effects:
            if applied_effect.effect == effect_func:
                return applied_effect.progress
        return 50  # Default to center for bidirectional sliders
    
    def save_applied_effect(self, effect_func: Callable, progress: int):
        """Save an applied effect"""
        # Remove existing effect of same type
        self.applied_effects = [e for e in self.applied_effects if e.effect != effect_func]
        # Add new effect
        if progress != 50:  # Only add if progress is not center (default)
            self.applied_effects.append(AppliedEffect(effect_func, progress))
    
    def undo(self):
        """Undo last applied effect"""
        if not self.applied_effects:
            return
        
        effect = self.applied_effects.pop()
        self.redo_stack.append(effect)
    
    def redo(self):
        """Redo last undone effect"""
        if not self.redo_stack:
            return
        
        effect = self.redo_stack.pop()
        # Remove any existing effect of same type
        self.applied_effects = [e for e in self.applied_effects if e.effect != effect.effect]
        self.applied_effects.append(effect)


class FaceParts:
    """Collection of face part manipulation functions with natural effects"""
    
    # EYES EFFECTS
    @staticmethod
    def eyes_magnified(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Magnify eyes with natural blending"""
        height, width = image.shape[:2]
        
        # Calculate strength - much more conservative
        strength = progress / 100.0 * 0.25  # Max 25% magnification
        
        # Left eye - landmarks 36-41
        left_eye_points = landmarks[36:42]
        left_eye_x = int(sum(p.x for p in left_eye_points) / len(left_eye_points) * width)
        left_eye_y = int(sum(p.y for p in left_eye_points) / len(left_eye_points) * height)
        left_radius = int(abs(landmarks[39].x - landmarks[36].x) * width * 1.8)
        
        # Right eye - landmarks 42-47
        right_eye_points = landmarks[42:48]
        right_eye_x = int(sum(p.x for p in right_eye_points) / len(right_eye_points) * width)
        right_eye_y = int(sum(p.y for p in right_eye_points) / len(right_eye_points) * height)
        right_radius = int(abs(landmarks[45].x - landmarks[42].x) * width * 1.8)
        
        # Use multi-point magnification for better blending
        eye_points = []
        if left_radius > 15:
            eye_points.append((left_eye_x, left_eye_y))
        if right_radius > 15:
            eye_points.append((right_eye_x, right_eye_y))
        
        if eye_points:
            avg_radius = max((left_radius + right_radius) // 2, 20)
            result = ImageEffects.multi_point_magnify(image, eye_points, avg_radius, strength)
        else:
            result = image
        
        return result
    
    @staticmethod
    def eyes_horizontal_distance(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Move eyes together or apart (50 = center, <50 = together, >50 = apart)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Calculate eye centers
        left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in range(36, 42)], axis=0)
        right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in range(42, 48)], axis=0)
        
        # Calculate the distance between eyes
        eye_distance = abs(right_eye_center[0] - left_eye_center[0]) * width
        shift_amount = eye_distance * shift_factor * 0.5  # Each eye moves half the distance
        
        # Left eye
        left_eye_x = int(left_eye_center[0] * width)
        left_eye_y = int(left_eye_center[1] * height)
        left_radius = int(abs(landmarks[39].x - landmarks[36].x) * width * 2.0)
        
        # Right eye
        right_eye_x = int(right_eye_center[0] * width)
        right_eye_y = int(right_eye_center[1] * height)
        right_radius = int(abs(landmarks[45].x - landmarks[42].x) * width * 2.0)
        
        # Apply shifts
        if abs(shift_amount) > 0.5:
            # Left eye moves right for together, left for apart
            result = ImageEffects.natural_shift(result, left_eye_x, left_eye_y, left_radius, shift_amount, 0)
            # Right eye moves left for together, right for apart
            result = ImageEffects.natural_shift(result, right_eye_x, right_eye_y, right_radius, -shift_amount, 0)
        
        return result
    
    @staticmethod
    def eyes_vertical_position(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Move eyes up or down (50 = center, <50 = up, >50 = down)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        shift_amount = shift_factor * 12  # Max 12 pixels
        
        # Process both eyes
        for eye_indices in [(36, 42), (42, 48)]:  # Left eye, right eye
            eye_points = landmarks[eye_indices[0]:eye_indices[1]]
            eye_x = int(sum(p.x for p in eye_points) / len(eye_points) * width)
            eye_y = int(sum(p.y for p in eye_points) / len(eye_points) * height)
            
            # Calculate radius
            if eye_indices[0] == 36:  # Left eye
                radius = int(abs(landmarks[39].x - landmarks[36].x) * width * 2.0)
            else:  # Right eye
                radius = int(abs(landmarks[45].x - landmarks[42].x) * width * 2.0)
            
            if radius > 15:
                result = ImageEffects.natural_shift(result, eye_x, eye_y, radius, 0, shift_amount)
        
        return result
    
    @staticmethod
    def eyes_vertical_stretch(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Stretch eyes vertically (50 = center, <50 = open more, >50 = close)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor
        stretch_factor = (50 - progress) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Process both eyes
        for eye_indices in [(36, 42), (42, 48)]:  # Left eye, right eye
            eye_points = landmarks[eye_indices[0]:eye_indices[1]]
            eye_x = int(sum(p.x for p in eye_points) / len(eye_points) * width)
            eye_y = int(sum(p.y for p in eye_points) / len(eye_points) * height)
            
            # Calculate radius
            if eye_indices[0] == 36:  # Left eye
                radius = int(abs(landmarks[39].x - landmarks[36].x) * width * 1.5)
            else:  # Right eye
                radius = int(abs(landmarks[45].x - landmarks[42].x) * width * 1.5)
            
            if radius > 15:
                result = ImageEffects.directional_stretch(result, eye_x, eye_y, radius, 0, stretch_factor)
        
        return result
    
    @staticmethod
    def eyes_horizontal_stretch(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Stretch eyes horizontally (50 = center, <50 = longer, >50 = shorter)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor
        stretch_factor = (50 - progress) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Process both eyes
        for eye_indices in [(36, 42), (42, 48)]:  # Left eye, right eye
            eye_points = landmarks[eye_indices[0]:eye_indices[1]]
            eye_x = int(sum(p.x for p in eye_points) / len(eye_points) * width)
            eye_y = int(sum(p.y for p in eye_points) / len(eye_points) * height)
            
            # Calculate radius
            if eye_indices[0] == 36:  # Left eye
                radius = int(abs(landmarks[39].x - landmarks[36].x) * width * 1.5)
            else:  # Right eye
                radius = int(abs(landmarks[45].x - landmarks[42].x) * width * 1.5)
            
            if radius > 15:
                result = ImageEffects.directional_stretch(result, eye_x, eye_y, radius, stretch_factor, 0)
        
        return result
    
    # EYEBROWS EFFECTS
    @staticmethod
    def eyebrows_asymmetric_vertical(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """One eyebrow up, other down (50 = center, <50 = left up/right down, >50 = opposite)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        shift_amount = shift_factor * 8  # Max 8 pixels
        
        # Left eyebrow (landmarks 17-21)
        left_brow_x = int(sum(landmarks[i].x for i in range(17, 22)) / 5 * width)
        left_brow_y = int(sum(landmarks[i].y for i in range(17, 22)) / 5 * height)
        left_radius = int(abs(landmarks[21].x - landmarks[17].x) * width * 0.8)
        
        # Right eyebrow (landmarks 22-26)
        right_brow_x = int(sum(landmarks[i].x for i in range(22, 27)) / 5 * width)
        right_brow_y = int(sum(landmarks[i].y for i in range(22, 27)) / 5 * height)
        right_radius = int(abs(landmarks[26].x - landmarks[22].x) * width * 0.8)
        
        # Apply opposite shifts
        if abs(shift_amount) > 0.5:
            result = ImageEffects.natural_shift(result, left_brow_x, left_brow_y, left_radius, 0, -shift_amount)
            result = ImageEffects.natural_shift(result, right_brow_x, right_brow_y, right_radius, 0, shift_amount)
        
        return result
    
    @staticmethod
    def eyebrows_symmetric_vertical(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Both eyebrows up or down (50 = center, <50 = up, >50 = down)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        shift_amount = shift_factor * 10  # Max 10 pixels
        
        # Process both eyebrows
        for brow_indices in [(17, 22), (22, 27)]:  # Left, right eyebrow
            brow_points = landmarks[brow_indices[0]:brow_indices[1]]
            brow_x = int(sum(p.x for p in brow_points) / len(brow_points) * width)
            brow_y = int(sum(p.y for p in brow_points) / len(brow_points) * height)
            
            # Calculate radius
            if brow_indices[0] == 17:  # Left eyebrow
                radius = int(abs(landmarks[21].x - landmarks[17].x) * width * 0.8)
            else:  # Right eyebrow
                radius = int(abs(landmarks[26].x - landmarks[22].x) * width * 0.8)
            
            if radius > 10:
                result = ImageEffects.natural_shift(result, brow_x, brow_y, radius, 0, shift_amount)
        
        return result
    
    @staticmethod
    def eyebrows_horizontal_extend(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Extend eyebrows horizontally (50 = center, <50 = extend outward, >50 = contract inward)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor
        stretch_factor = (50 - progress) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Process both eyebrows
        for brow_indices in [(17, 22), (22, 27)]:  # Left, right eyebrow
            brow_points = landmarks[brow_indices[0]:brow_indices[1]]
            brow_x = int(sum(p.x for p in brow_points) / len(brow_points) * width)
            brow_y = int(sum(p.y for p in brow_points) / len(brow_points) * height)
            
            # Calculate radius
            if brow_indices[0] == 17:  # Left eyebrow
                radius = int(abs(landmarks[21].x - landmarks[17].x) * width * 0.6)
            else:  # Right eyebrow
                radius = int(abs(landmarks[26].x - landmarks[22].x) * width * 0.6)
            
            if radius > 10:
                result = ImageEffects.directional_stretch(result, brow_x, brow_y, radius, stretch_factor, 0)
        
        return result
    
    @staticmethod
    def eyebrows_rotation(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Rotate eyebrows (50 = center, <50 = left CCW/right CW, >50 = opposite)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to rotation angle
        rotation_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        rotation_angle = rotation_factor * 15  # Max 15 degrees
        
        # Left eyebrow
        left_brow_x = int(sum(landmarks[i].x for i in range(17, 22)) / 5 * width)
        left_brow_y = int(sum(landmarks[i].y for i in range(17, 22)) / 5 * height)
        left_radius = int(abs(landmarks[21].x - landmarks[17].x) * width * 0.6)
        
        # Right eyebrow
        right_brow_x = int(sum(landmarks[i].x for i in range(22, 27)) / 5 * width)
        right_brow_y = int(sum(landmarks[i].y for i in range(22, 27)) / 5 * height)
        right_radius = int(abs(landmarks[26].x - landmarks[22].x) * width * 0.6)
        
        # Apply opposite rotations
        if abs(rotation_angle) > 0.5:
            result = ImageEffects.rotate_region(result, left_brow_x, left_brow_y, left_radius, -rotation_angle)
            result = ImageEffects.rotate_region(result, right_brow_x, right_brow_y, right_radius, rotation_angle)
        
        return result
    
    @staticmethod
    def eyebrows_thickness(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Adjust eyebrow thickness (50 = normal, <50 = thinner, >50 = thicker)"""
        height, width = image.shape[:2]
        
        # Convert progress to strength
        if progress < 50:
            # Thinning - use slight vertical compression
            strength = (50 - progress) / 50.0 * 0.2
            stretch_y = strength  # Compress vertically
        else:
            # Thickening - use slight magnification
            strength = (progress - 50) / 50.0 * 0.15
            
            # Process both eyebrows with magnification
            brow_points = []
            
            # Left eyebrow center
            left_brow_x = int(sum(landmarks[i].x for i in range(17, 22)) / 5 * width)
            left_brow_y = int(sum(landmarks[i].y for i in range(17, 22)) / 5 * height)
            brow_points.append((left_brow_x, left_brow_y))
            
            # Right eyebrow center
            right_brow_x = int(sum(landmarks[i].x for i in range(22, 27)) / 5 * width)
            right_brow_y = int(sum(landmarks[i].y for i in range(22, 27)) / 5 * height)
            brow_points.append((right_brow_x, right_brow_y))
            
            avg_radius = int(abs(landmarks[26].x - landmarks[17].x) * width * 0.2)
            return ImageEffects.multi_point_magnify(image, brow_points, avg_radius, strength)
        
        # For thinning, apply vertical compression
        result = image.copy()
        for brow_indices in [(17, 22), (22, 27)]:
            brow_points = landmarks[brow_indices[0]:brow_indices[1]]
            brow_x = int(sum(p.x for p in brow_points) / len(brow_points) * width)
            brow_y = int(sum(p.y for p in brow_points) / len(brow_points) * height)
            
            if brow_indices[0] == 17:
                radius = int(abs(landmarks[21].x - landmarks[17].x) * width * 0.5)
            else:
                radius = int(abs(landmarks[26].x - landmarks[22].x) * width * 0.5)
            
            result = ImageEffects.directional_stretch(result, brow_x, brow_y, radius, 0, stretch_y)
        
        return result
    
    # NOSE EFFECTS
    @staticmethod
    def nose_magnified(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Magnify nose naturally"""
        height, width = image.shape[:2]
        
        # Calculate strength
        strength = progress / 100.0 * 0.45
        
        # Nose center (landmark 30)
        nose_x = int(landmarks[30].x * width)
        nose_y = int(landmarks[30].y * height)
        
        # Calculate radius based on nose features
        nose_width = abs(landmarks[35].x - landmarks[31].x) * width
        nose_height = abs(landmarks[27].y - landmarks[33].y) * height
        radius = max(int((nose_width + nose_height) * 0.4), 18)
        
        result = ImageEffects.natural_magnify(image, nose_x, nose_y, radius, strength)
        
        return result
    
    @staticmethod
    def nose_vertical_position(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Move nose up or down (50 = center, <50 = up, >50 = down)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        shift_amount = shift_factor * 15  # Max 15 pixels
        
        # Nose center
        nose_x = int(landmarks[30].x * width)
        nose_y = int(landmarks[30].y * height)
        
        # Calculate radius to include entire nose
        nose_width = abs(landmarks[35].x - landmarks[31].x) * width
        nose_height = abs(landmarks[27].y - landmarks[33].y) * height
        radius = max(int((nose_width + nose_height) * 0.5), 25)
        
        if abs(shift_amount) > 0.5:
            result = ImageEffects.natural_shift(result, nose_x, nose_y, radius, 0, shift_amount)
        
        return result
    
    @staticmethod
    def nose_upper_width(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Adjust upper nose width (50 = center, <50 = wider, >50 = narrower)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor
        stretch_factor = (50 - progress) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Upper nose area (around landmark 27-28)
        upper_nose_x = int(landmarks[27].x * width)
        upper_nose_y = int((landmarks[27].y + landmarks[28].y) / 2 * height)
        
        # Calculate radius
        nose_width = abs(landmarks[35].x - landmarks[31].x) * width
        radius = int(nose_width * 0.8)
        
        if radius > 10:
            result = ImageEffects.directional_stretch(result, upper_nose_x, upper_nose_y, radius, stretch_factor, 0)
        
        return result
    
    @staticmethod
    def nose_nostril_width(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Adjust nostril width (50 = center, <50 = narrower, >50 = wider)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor (opposite of upper nose)
        stretch_factor = (progress - 50) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Nostril area (around landmarks 31-35)
        nostril_x = int((landmarks[31].x + landmarks[35].x) / 2 * width)
        nostril_y = int((landmarks[31].y + landmarks[35].y) / 2 * height)
        
        # Calculate radius
        nostril_width = abs(landmarks[35].x - landmarks[31].x) * width
        radius = int(nostril_width * 1.2)
        
        if radius > 10:
            result = ImageEffects.directional_stretch(result, nostril_x, nostril_y, radius, stretch_factor, 0)
        
        return result
    
    @staticmethod
    def nose_nostril_vertical(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Move nostrils up or down (50 = center, <50 = down, >50 = up)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (50 - progress) / 50.0  # -1.0 to +1.0 (inverted)
        shift_amount = shift_factor * 8  # Max 8 pixels
        
        # Nostril area
        nostril_x = int((landmarks[31].x + landmarks[35].x) / 2 * width)
        nostril_y = int((landmarks[31].y + landmarks[35].y) / 2 * height)
        
        # Smaller radius for just the nostril area
        nostril_width = abs(landmarks[35].x - landmarks[31].x) * width
        radius = int(nostril_width * 0.8)
        
        if abs(shift_amount) > 0.5:
            result = ImageEffects.natural_shift(result, nostril_x, nostril_y, radius, 0, shift_amount)
        
        return result
    
    # LIPS EFFECTS
    @staticmethod
    def lips_magnified(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Magnify entire lips area naturally"""
        height, width = image.shape[:2]
        
        # Calculate strength
        strength = progress / 100.0 * 0.35  # Increased for more noticeable effect
        
        # Calculate lip dimensions
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_top = landmarks[51]
        mouth_bottom = landmarks[57]
        
        # Lip center
        mouth_center_x = int((mouth_left.x + mouth_right.x) / 2 * width)
        mouth_center_y = int((mouth_top.y + mouth_bottom.y) / 2 * height)
        
        # Calculate elliptical radii for natural lip shape
        mouth_width = abs(mouth_right.x - mouth_left.x) * width
        mouth_height = abs(mouth_bottom.y - mouth_top.y) * height
        
        # Use elliptical magnification for more natural effect
        width_radius = int(mouth_width * 0.8)
        height_radius = int(mouth_height * 1.5)
        
        result = ImageEffects.elliptical_magnify(image, mouth_center_x, mouth_center_y, 
                                                 width_radius, height_radius, strength)
        
        return result
    
    @staticmethod
    def lips_vertical_stretch(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Stretch lips vertically (50 = center, <50 = upper up/lower down, >50 = opposite)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor
        stretch_factor = (50 - progress) / 50.0 * 0.35  # -0.35 to +0.35
        
        # Upper lip center (landmarks 48-54, focusing on top curve)
        upper_lip_x = int((landmarks[50].x + landmarks[51].x + landmarks[52].x) / 3 * width)
        upper_lip_y = int((landmarks[50].y + landmarks[51].y + landmarks[52].y) / 3 * height)
        
        # Lower lip center (landmarks 54-60, focusing on bottom curve)
        lower_lip_x = int((landmarks[56].x + landmarks[57].x + landmarks[58].x) / 3 * width)
        lower_lip_y = int((landmarks[56].y + landmarks[57].y + landmarks[58].y) / 3 * height)
        
        # Calculate radius based on lip size
        lip_width = abs(landmarks[54].x - landmarks[48].x) * width
        radius = int(lip_width * 0.5)
        
        if abs(stretch_factor) > 0.01:
            # Upper lip - negative shift moves up
            result = ImageEffects.natural_shift(result, upper_lip_x, upper_lip_y, radius, 0, -stretch_factor * 10)
            # Lower lip - positive shift moves down
            result = ImageEffects.natural_shift(result, lower_lip_x, lower_lip_y, radius, 0, stretch_factor * 10)
        
        return result
    
    @staticmethod
    def lips_horizontal_stretch(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Stretch lips horizontally (50 = center, <50 = outward, >50 = inward)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to stretch factor
        stretch_factor = (50 - progress) / 50.0 * 0.3  # -0.3 to +0.3
        
        # Left corner of lips
        left_corner_x = int(landmarks[48].x * width)
        left_corner_y = int(landmarks[48].y * height)
        
        # Right corner of lips
        right_corner_x = int(landmarks[54].x * width)
        right_corner_y = int(landmarks[54].y * height)
        
        # Calculate radius
        lip_width = abs(landmarks[54].x - landmarks[48].x) * width
        radius = int(lip_width * 0.4)
        
        if abs(stretch_factor) > 0.01:
            # Left corner - negative shift moves left (outward)
            result = ImageEffects.natural_shift(result, left_corner_x, left_corner_y, radius, -stretch_factor * 12, 0)
            # Right corner - positive shift moves right (outward)
            result = ImageEffects.natural_shift(result, right_corner_x, right_corner_y, radius, stretch_factor * 12, 0)
        
        return result
    
    @staticmethod
    def lips_vertical_position(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Move entire lips up or down (50 = center, <50 = up, >50 = down)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        shift_amount = shift_factor * 15  # Max 15 pixels
        
        # Lip center (using all lip landmarks)
        lip_indices = list(range(48, 60))  # All lip landmarks
        lip_x = int(sum(landmarks[i].x for i in lip_indices) / len(lip_indices) * width)
        lip_y = int(sum(landmarks[i].y for i in lip_indices) / len(lip_indices) * height)
        
        # Calculate radius to include entire lip area
        lip_width = abs(landmarks[54].x - landmarks[48].x) * width
        lip_height = abs(landmarks[57].y - landmarks[51].y) * height
        radius = int(max(lip_width, lip_height) * 0.8)
        
        if abs(shift_amount) > 0.5:
            result = ImageEffects.natural_shift(result, lip_x, lip_y, radius, 0, shift_amount)
        
        return result
    
    # FACE EFFECTS
    @staticmethod
    def face_slim(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Slim the face by compressing cheek and jaw areas"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Calculate strength
        strength = progress / 100.0 * 0.4
        
        # Get face outline points for proper slimming areas
        # Left side of face (jaw/cheek area)
        left_jaw_points = [landmarks[i] for i in [1, 2, 3, 4, 5]]
        left_center_x = int(sum(p.x for p in left_jaw_points) / len(left_jaw_points) * width)
        left_center_y = int(sum(p.y for p in left_jaw_points) / len(left_jaw_points) * height)
        
        # Right side of face (jaw/cheek area)
        right_jaw_points = [landmarks[i] for i in [11, 12, 13, 14, 15]]
        right_center_x = int(sum(p.x for p in right_jaw_points) / len(right_jaw_points) * width)
        right_center_y = int(sum(p.y for p in right_jaw_points) / len(right_jaw_points) * height)
        
        # Calculate radius based on face width
        face_width = abs(landmarks[16].x - landmarks[0].x) * width
        radius = int(face_width * 0.2)
        
        # Shift left side inward
        inward_shift = strength * 15
        result = ImageEffects.natural_shift(result, left_center_x, left_center_y, radius, inward_shift, 0)
        
        # Shift right side inward
        result = ImageEffects.natural_shift(result, right_center_x, right_center_y, radius, -inward_shift, 0)
        
        # Also slim the upper cheek areas
        left_cheek_x = int(landmarks[2].x * width)
        left_cheek_y = int(landmarks[2].y * height)
        right_cheek_x = int(landmarks[14].x * width)
        right_cheek_y = int(landmarks[14].y * height)
        
        upper_radius = int(face_width * 0.15)
        upper_shift = strength * 8
        
        result = ImageEffects.natural_shift(result, left_cheek_x, left_cheek_y, upper_radius, upper_shift, 0)
        result = ImageEffects.natural_shift(result, right_cheek_x, right_cheek_y, upper_radius, -upper_shift, 0)
        
        return result
    
    @staticmethod
    def face_chin_fat(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Adjust chin fat (50 = normal, <50 = less fat, >50 = more fat)"""
        height, width = image.shape[:2]
        
        # Calculate chin and jaw area more comprehensively
        # Use landmarks 3-13 for broader chin/jaw area
        chin_jaw_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        chin_jaw_points = [landmarks[i] for i in chin_jaw_indices]
        
        # Center point for chin fat (lower part of face)
        chin_x = int(sum(p.x for p in chin_jaw_points) / len(chin_jaw_points) * width)
        chin_y = int(sum(p.y for p in chin_jaw_points) / len(chin_jaw_points) * height)
        
        # Move center slightly down for better effect
        chin_y += int(0.02 * height)
        
        # Calculate effect based on progress
        if progress < 50:
            # Reduce chin fat - use stronger compression
            strength = (50 - progress) / 50.0 * 0.4  # Increased from 0.25
            
            # Apply vertical compression to slim the chin
            radius = int(abs(landmarks[16].x - landmarks[0].x) * width * 0.35)  # Larger radius
            result = ImageEffects.directional_stretch(image, chin_x, chin_y, radius, 0, strength)
            
            # Additional compression at the jawline
            left_jaw_x = int(landmarks[4].x * width)
            left_jaw_y = int(landmarks[4].y * height)
            right_jaw_x = int(landmarks[12].x * width)
            right_jaw_y = int(landmarks[12].y * height)
            
            jaw_radius = int(abs(landmarks[16].x - landmarks[0].x) * width * 0.15)
            result = ImageEffects.directional_stretch(result, left_jaw_x, left_jaw_y, jaw_radius, strength * 0.3, strength * 0.5)
            result = ImageEffects.directional_stretch(result, right_jaw_x, right_jaw_y, jaw_radius, -strength * 0.3, strength * 0.5)
        else:
            # Increase chin fat - use stronger magnification
            strength = (progress - 50) / 50.0 * 0.35  # Increased from 0.2
            
            # Apply magnification to expand chin area
            radius = int(abs(landmarks[16].x - landmarks[0].x) * width * 0.4)  # Larger radius
            result = ImageEffects.natural_magnify(image, chin_x, chin_y, radius, strength)
            
            # Additional magnification at the lower jaw
            lower_chin_y = chin_y + int(0.03 * height)
            result = ImageEffects.natural_magnify(result, chin_x, lower_chin_y, int(radius * 0.7), strength * 0.6)
        
        return result
    
    @staticmethod
    def face_chin_position(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Move chin up or down (50 = center, <50 = up, >50 = down)"""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Convert progress to shift amount
        shift_factor = (progress - 50) / 50.0  # -1.0 to +1.0
        shift_amount = shift_factor * 12  # Max 12 pixels
        
        # Chin area
        chin_x = int(landmarks[8].x * width)
        chin_y = int(landmarks[8].y * height)
        
        # Include lower face area
        radius = int(abs(landmarks[16].x - landmarks[0].x) * width * 0.3)
        
        if abs(shift_amount) > 0.5:
            result = ImageEffects.natural_shift(result, chin_x, chin_y, radius, 0, shift_amount)
        
        return result
    
    @staticmethod
    def face_forehead_size(landmarks: List[NormalizedLandmark], image: np.ndarray, progress: int) -> np.ndarray:
        """Adjust forehead size (50 = normal, <50 = smaller, >50 = larger)"""
        height, width = image.shape[:2]
        
        # Forehead area (above eyebrows)
        forehead_x = int((landmarks[19].x + landmarks[24].x) / 2 * width)
        # Estimate forehead position above eyebrows
        eyebrow_y = (landmarks[19].y + landmarks[24].y) / 2
        forehead_y = int((eyebrow_y - 0.1) * height)  # 10% above eyebrows
        
        # Calculate effect based on progress
        if progress < 50:
            # Smaller forehead - shift down
            shift_factor = (50 - progress) / 50.0 * 0.3
            shift_amount = shift_factor * 15  # Max 15 pixels down
            
            radius = int(abs(landmarks[26].x - landmarks[17].x) * width * 0.8)
            result = ImageEffects.natural_shift(image, forehead_x, forehead_y, radius, 0, shift_amount)
        else:
            # Larger forehead - use vertical stretch
            strength = (progress - 50) / 50.0 * 0.2
            stretch_y = -strength  # Negative to expand upward
            
            radius = int(abs(landmarks[26].x - landmarks[17].x) * width * 0.8)
            result = ImageEffects.directional_stretch(image, forehead_x, forehead_y, radius, 0, stretch_y)
        
        return result


class FaceRetouchSystem:
    """Main face retouching system"""
    
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.setup_predictor()
    
    def setup_predictor(self):
        """Setup dlib face landmark predictor"""
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        
        # Check if predictor file exists
        import os
        if not os.path.exists(predictor_path):
            print("Predictor file not found. Downloading...")
            self.download_predictor(predictor_path)
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            print("Face predictor loaded successfully!")
        except Exception as e:
            print(f"Error loading face predictor: {e}")
            print("Please manually download shape_predictor_68_face_landmarks.dat")
    
    def download_predictor(self, predictor_path):
        """Download the face predictor file"""
        try:
            import urllib.request
            import bz2
            
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
            
            print("Downloading predictor file (this may take a while)...")
            urllib.request.urlretrieve(url, compressed_file)
            
            print("Extracting predictor file...")
            with bz2.BZ2File(compressed_file, 'rb') as f_in:
                with open(predictor_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Clean up compressed file
            import os
            os.remove(compressed_file)
            print("Predictor file downloaded and extracted successfully!")
            
        except Exception as e:
            print(f"Error downloading predictor: {e}")
            print("Please manually download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract the .dat file to your project directory")
    
    def detect_faces(self, image: np.ndarray) -> List[Landmarks]:
        """Detect faces and extract landmarks"""
        if self.predictor is None:
            raise ValueError("Face predictor not loaded")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        landmarks_list = []
        
        for face in faces:
            # Get face coordinates
            face_coords = FaceCoordinates(face.left(), face.top(), face.right(), face.bottom())
            
            # Get landmarks
            landmarks = self.predictor(gray, face)
            normalized_landmarks = []
            
            for i in range(68):  # 68 landmarks in dlib model
                point = landmarks.part(i)
                normalized_landmarks.append(
                    NormalizedLandmark(
                        x=point.x / image.shape[1],
                        y=point.y / image.shape[0],
                        z=0.0
                    )
                )
            
            # Calculate face angle (simplified)
            angle = self.calculate_face_angle(normalized_landmarks)
            
            landmarks_list.append(Landmarks(angle, face_coords, normalized_landmarks))
        
        return landmarks_list
    
    def calculate_face_angle(self, landmarks: List[NormalizedLandmark]) -> float:
        """Calculate face rotation angle"""
        # Use eye positions to calculate angle
        left_eye = [landmarks[36].x, landmarks[36].y]
        right_eye = [landmarks[45].x, landmarks[45].y]
        
        angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        return angle
    
    def apply_all_effects(self, image: np.ndarray, selected_face: Landmarks) -> np.ndarray:
        """Apply all saved effects to the image"""
        result = image.copy()
        
        for effect in selected_face.applied_effects:
            try:
                result = effect.effect(selected_face.normalized_landmarks, result, effect.progress)
            except Exception as e:
                print(f"Error applying saved effect: {e}")
        
        return result


# Enhanced GUI Interface with sections
class FaceRetouchGUI:
    """Enhanced GUI interface for face retouching with sections"""
    
    def __init__(self, image_path: str):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.retouch_system = FaceRetouchSystem()
        if self.retouch_system.predictor is None:
            raise ValueError("Face predictor not loaded")
        
        # Detect faces
        self.faces = self.retouch_system.detect_faces(self.original_image)
        if not self.faces:
            raise ValueError("No faces detected in image")
        
        self.selected_face = self.faces[0]  # Use first face
        self.current_image = self.original_image.copy()
        
        # Current section
        self.current_section = 'eyes'  # Default section
        
        # Effect parameters - all start at 50 (center) for bidirectional control
        self.effects = {
            'eyes': {
                'magnify': 0,  # 0-100 (unidirectional)
                'horizontal_distance': 50,  # 0-100 (50 = center)
                'vertical_position': 50,
                'vertical_stretch': 50,
                'horizontal_stretch': 50
            },
            'eyebrows': {
                'asymmetric_vertical': 50,
                'symmetric_vertical': 50,
                'horizontal_extend': 50,
                'rotation': 50,
                'thickness': 50
            },
            'nose': {
                'magnify': 0,  # 0-100 (unidirectional)
                'vertical_position': 50,
                'upper_width': 50,
                'nostril_width': 50,
                'nostril_vertical': 50
            },
            'lips': {
                'magnify': 0,  # 0-100 (unidirectional)
                'vertical_stretch': 50,
                'horizontal_stretch': 50,
                'vertical_position': 50
            },
            'face': {
                'slim': 0,  # 0-100 (unidirectional)
                'chin_fat': 50,
                'chin_position': 50,
                'forehead_size': 50
            }
        }
        
        # Effect functions mapping
        self.effect_functions = {
            'eyes': {
                'magnify': FaceParts.eyes_magnified,
                'horizontal_distance': FaceParts.eyes_horizontal_distance,
                'vertical_position': FaceParts.eyes_vertical_position,
                'vertical_stretch': FaceParts.eyes_vertical_stretch,
                'horizontal_stretch': FaceParts.eyes_horizontal_stretch
            },
            'eyebrows': {
                'asymmetric_vertical': FaceParts.eyebrows_asymmetric_vertical,
                'symmetric_vertical': FaceParts.eyebrows_symmetric_vertical,
                'horizontal_extend': FaceParts.eyebrows_horizontal_extend,
                'rotation': FaceParts.eyebrows_rotation,
                'thickness': FaceParts.eyebrows_thickness
            },
            'nose': {
                'magnify': FaceParts.nose_magnified,
                'vertical_position': FaceParts.nose_vertical_position,
                'upper_width': FaceParts.nose_upper_width,
                'nostril_width': FaceParts.nose_nostril_width,
                'nostril_vertical': FaceParts.nose_nostril_vertical
            },
            'lips': {
                'magnify': FaceParts.lips_magnified,
                'vertical_stretch': FaceParts.lips_vertical_stretch,
                'horizontal_stretch': FaceParts.lips_horizontal_stretch,
                'vertical_position': FaceParts.lips_vertical_position
            },
            'face': {
                'slim': FaceParts.face_slim,
                'chin_fat': FaceParts.face_chin_fat,
                'chin_position': FaceParts.face_chin_position,
                'forehead_size': FaceParts.face_forehead_size
            }
        }
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI with sections and trackbars"""
        cv2.namedWindow('Face Retouch', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Section Selector', cv2.WINDOW_AUTOSIZE)
        
        # Create section selector
        self.create_section_selector()
        
        # Create trackbars for current section
        self.update_trackbars()
        
        # Update display
        self.update_display()
    
    def create_section_selector(self):
        """Create section selector panel"""
        selector = np.ones((200, 300, 3), dtype=np.uint8) * 240
        
        sections = ['Eyes', 'Eyebrows', 'Nose', 'Lips', 'Face']
        colors = {
            'eyes': (255, 0, 0),
            'eyebrows': (0, 255, 0),
            'nose': (0, 0, 255),
            'lips': (255, 0, 255),
            'face': (0, 255, 255)
        }
        
        cv2.putText(selector, 'Select Section:', (50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        for i, section in enumerate(sections):
            y_pos = 60 + i * 25
            section_key = section.lower()
            color = (0, 0, 0) if section_key != self.current_section else colors[section_key]
            thickness = 1 if section_key != self.current_section else 2
            
            cv2.putText(selector, f'{i+1}. {section}', (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        cv2.putText(selector, 'Press 1-5 to switch', (50, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow('Section Selector', selector)
    
    def update_trackbars(self):
        """Update trackbars for current section"""
        # Clear existing trackbars
        cv2.destroyWindow('Controls')
        cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
        
        # Create trackbars based on current section
        if self.current_section == 'eyes':
            cv2.createTrackbar('Magnify', 'Controls', 
                             self.effects['eyes']['magnify'], 100, 
                             lambda v: self.on_effect_change('magnify', v))
            cv2.createTrackbar('Distance (Together/Apart)', 'Controls', 
                             self.effects['eyes']['horizontal_distance'], 100, 
                             lambda v: self.on_effect_change('horizontal_distance', v))
            cv2.createTrackbar('Position (Up/Down)', 'Controls', 
                             self.effects['eyes']['vertical_position'], 100, 
                             lambda v: self.on_effect_change('vertical_position', v))
            cv2.createTrackbar('Open/Close', 'Controls', 
                             self.effects['eyes']['vertical_stretch'], 100, 
                             lambda v: self.on_effect_change('vertical_stretch', v))
            cv2.createTrackbar('Length', 'Controls', 
                             self.effects['eyes']['horizontal_stretch'], 100, 
                             lambda v: self.on_effect_change('horizontal_stretch', v))
        
        elif self.current_section == 'eyebrows':
            cv2.createTrackbar('Asymmetric Up/Down', 'Controls', 
                             self.effects['eyebrows']['asymmetric_vertical'], 100, 
                             lambda v: self.on_effect_change('asymmetric_vertical', v))
            cv2.createTrackbar('Both Up/Down', 'Controls', 
                             self.effects['eyebrows']['symmetric_vertical'], 100, 
                             lambda v: self.on_effect_change('symmetric_vertical', v))
            cv2.createTrackbar('Extend/Contract', 'Controls', 
                             self.effects['eyebrows']['horizontal_extend'], 100, 
                             lambda v: self.on_effect_change('horizontal_extend', v))
            cv2.createTrackbar('Rotation', 'Controls', 
                             self.effects['eyebrows']['rotation'], 100, 
                             lambda v: self.on_effect_change('rotation', v))
            cv2.createTrackbar('Thickness', 'Controls', 
                             self.effects['eyebrows']['thickness'], 100, 
                             lambda v: self.on_effect_change('thickness', v))
        
        elif self.current_section == 'nose':
            cv2.createTrackbar('Magnify', 'Controls', 
                             self.effects['nose']['magnify'], 100, 
                             lambda v: self.on_effect_change('magnify', v))
            cv2.createTrackbar('Position (Up/Down)', 'Controls', 
                             self.effects['nose']['vertical_position'], 100, 
                             lambda v: self.on_effect_change('vertical_position', v))
            cv2.createTrackbar('Upper Width', 'Controls', 
                             self.effects['nose']['upper_width'], 100, 
                             lambda v: self.on_effect_change('upper_width', v))
            cv2.createTrackbar('Nostril Width', 'Controls', 
                             self.effects['nose']['nostril_width'], 100, 
                             lambda v: self.on_effect_change('nostril_width', v))
            cv2.createTrackbar('Nostril Position', 'Controls', 
                             self.effects['nose']['nostril_vertical'], 100, 
                             lambda v: self.on_effect_change('nostril_vertical', v))
        
        elif self.current_section == 'lips':
            cv2.createTrackbar('Magnify', 'Controls', 
                             self.effects['lips']['magnify'], 100, 
                             lambda v: self.on_effect_change('magnify', v))
            cv2.createTrackbar('Vertical Stretch', 'Controls', 
                             self.effects['lips']['vertical_stretch'], 100, 
                             lambda v: self.on_effect_change('vertical_stretch', v))
            cv2.createTrackbar('Horizontal Stretch', 'Controls', 
                             self.effects['lips']['horizontal_stretch'], 100, 
                             lambda v: self.on_effect_change('horizontal_stretch', v))
            cv2.createTrackbar('Position (Up/Down)', 'Controls', 
                             self.effects['lips']['vertical_position'], 100, 
                             lambda v: self.on_effect_change('vertical_position', v))
        
        elif self.current_section == 'face':
            cv2.createTrackbar('Slim', 'Controls', 
                             self.effects['face']['slim'], 100, 
                             lambda v: self.on_effect_change('slim', v))
            cv2.createTrackbar('Chin Fat', 'Controls', 
                             self.effects['face']['chin_fat'], 100, 
                             lambda v: self.on_effect_change('chin_fat', v))
            cv2.createTrackbar('Chin Position', 'Controls', 
                             self.effects['face']['chin_position'], 100, 
                             lambda v: self.on_effect_change('chin_position', v))
            cv2.createTrackbar('Forehead Size', 'Controls', 
                             self.effects['face']['forehead_size'], 100, 
                             lambda v: self.on_effect_change('forehead_size', v))
        
        # Update control panel
        self.update_control_panel()
    
    def update_control_panel(self):
        """Update control panel display"""
        panel_height = 300
        control_panel = np.ones((panel_height, 500, 3), dtype=np.uint8) * 240
        
        cv2.putText(control_panel, f'{self.current_section.title()} Controls', (50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add instructions based on section
        y_offset = 70
        instructions = {
            'eyes': [
                'Magnify: Enlarge eyes (0-100)',
                'Distance: Together(0) <-> Apart(100)',
                'Position: Up(0) <-> Down(100)',
                'Open/Close: Open(0) <-> Close(100)',
                'Length: Longer(0) <-> Shorter(100)'
            ],
            'eyebrows': [
                'Asymmetric: L↑R↓(0) <-> L↓R↑(100)',
                'Both: Up(0) <-> Down(100)',
                'Extend: Outward(0) <-> Inward(100)',
                'Rotation: CCW/CW(0) <-> CW/CCW(100)',
                'Thickness: Thin(0) <-> Thick(100)'
            ],
            'nose': [
                'Magnify: Enlarge nose (0-100)',
                'Position: Up(0) <-> Down(100)',
                'Upper Width: Wide(0) <-> Narrow(100)',
                'Nostril Width: Narrow(0) <-> Wide(100)',
                'Nostril Pos: Down(0) <-> Up(100)'
            ],
            'lips': [
                'Magnify: Enlarge lips (0-100)',
                'Vertical Stretch: Apart(0) <-> Together(100)',
                'Horizontal Stretch: Wider(0) <-> Narrower(100)',
                'Position: Up(0) <-> Down(100)'
            ],
            'face': [
                'Slim: Reduce face width (0-100)',
                'Chin Fat: Less(0) <-> More(100)',
                'Chin Position: Up(0) <-> Down(100)',
                'Forehead: Smaller(0) <-> Larger(100)'
            ]
        }
        
        for instruction in instructions.get(self.current_section, []):
            cv2.putText(control_panel, instruction, (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
        
        # Add general instructions
        cv2.putText(control_panel, 'Press 1-5 to switch sections', (50, panel_height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(control_panel, 'S: Save | R: Reset | Q: Quit', (50, panel_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow('Controls', control_panel)
    
    def on_effect_change(self, effect_name: str, value: int):
        """Handle effect change"""
        self.effects[self.current_section][effect_name] = value
        self.update_effects()
    
    def update_effects(self):
        """Apply all effects and update display"""
        try:
            # Start with original image
            self.current_image = self.original_image.copy()
            
            # Apply all effects from all sections
            for section, section_effects in self.effects.items():
                for effect_name, value in section_effects.items():
                    # Skip if value is at default
                    if (effect_name in ['magnify', 'slim'] and value == 0) or \
                       (effect_name not in ['magnify', 'slim'] and value == 50):
                        continue
                    
                    # Get effect function
                    effect_func = self.effect_functions[section].get(effect_name)
                    if effect_func:
                        self.current_image = effect_func(
                            self.selected_face.normalized_landmarks,
                            self.current_image,
                            value
                        )
            
            self.update_display()
            
        except Exception as e:
            print(f"Error applying effects: {e}")
    
    def update_display(self):
        """Update the main display window"""
        # Resize image if too large
        display_image = self.current_image.copy()
        height, width = display_image.shape[:2]
        
        if width > 800 or height > 600:
            scale = min(800/width, 600/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        cv2.imshow('Face Retouch', display_image)
    
    def reset_section(self):
        """Reset current section to default values"""
        for effect_name in self.effects[self.current_section]:
            if effect_name in ['magnify', 'slim']:
                self.effects[self.current_section][effect_name] = 0
            else:
                self.effects[self.current_section][effect_name] = 50
        
        self.update_trackbars()
        self.update_effects()
    
    def reset_all(self):
        """Reset all effects to default values"""
        for section in self.effects:
            for effect_name in self.effects[section]:
                if effect_name in ['magnify', 'slim']:
                    self.effects[section][effect_name] = 0
                else:
                    self.effects[section][effect_name] = 50
        
        self.update_trackbars()
        self.update_effects()
    
    def run(self):
        """Run the GUI"""
        print("\nEnhanced Face Retouching System")
        print("=" * 40)
        print("Press 1-5 to switch between sections:")
        print("1: Eyes | 2: Eyebrows | 3: Nose | 4: Lips | 5: Face")
        print("Use sliders to adjust effects")
        print("S: Save | R: Reset current | Shift+R: Reset all | Q: Quit")
        print("=" * 40)
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                output_path = 'enhanced_retouched_output.jpg'
                cv2.imwrite(output_path, self.current_image)
                print(f"Image saved as {output_path}")
            elif key == ord('r'):
                self.reset_section()
                print(f"Reset {self.current_section} section")
            elif key == ord('R'):  # Shift+R
                self.reset_all()
                print("Reset all sections")
            elif key == ord('1'):
                self.current_section = 'eyes'
                self.create_section_selector()
                self.update_trackbars()
            elif key == ord('2'):
                self.current_section = 'eyebrows'
                self.create_section_selector()
                self.update_trackbars()
            elif key == ord('3'):
                self.current_section = 'nose'
                self.create_section_selector()
                self.update_trackbars()
            elif key == ord('4'):
                self.current_section = 'lips'
                self.create_section_selector()
                self.update_trackbars()
            elif key == ord('5'):
                self.current_section = 'face'
                self.create_section_selector()
                self.update_trackbars()
        
        cv2.destroyAllWindows()


def run_enhanced_gui(image_path):
    """Run the enhanced GUI with an image"""
    try:
        gui = FaceRetouchGUI(image_path)
        gui.run()
        return True
    except Exception as e:
        print(f"Error running GUI: {e}")
        return False


def main():
    """Main function"""
    print("Enhanced Face Retouching System")
    print("=" * 50)
    
    import sys
    import os
    
    # Check if image path provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"Loading image: {image_path}")
            success = run_enhanced_gui(image_path)
            if success:
                return
        else:
            print(f"Image file not found: {image_path}")
    
    # Try to find image files in current directory
    common_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    current_dir = os.getcwd()
    
    image_files = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(ext) for ext in common_extensions):
            image_files.append(file)
    
    if image_files:
        print(f"\nFound {len(image_files)} image file(s):")
        for i, file in enumerate(image_files):
            print(f"{i+1}. {file}")
        
        try:
            choice = int(input(f"Select image (1-{len(image_files)}): ")) - 1
            if 0 <= choice < len(image_files):
                selected_file = image_files[choice]
                print(f"Loading: {selected_file}")
                success = run_enhanced_gui(selected_file)
                if success:
                    return
            else:
                print("Invalid selection")
        except (ValueError, KeyboardInterrupt):
            print("\nOperation cancelled")
    else:
        print("No image files found in current directory")
        print("Usage: python face_retouch.py <image_path>")


if __name__ == "__main__":
    main()