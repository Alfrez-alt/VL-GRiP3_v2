"""RGB-D capture for the Orbbec Gemini 2L camera (via the pyorbbecsdk).

Drop-in replacement for the previous RealSense grabber: it streams aligned
color + depth, lets the user preview the scene, and on key 's' saves
``rgb.png`` (BGR, 8-bit) and ``depth.npy`` (float32, in metres) into the sample
directory expected by the rest of the pipeline.

NOTE (validate on hardware): the exact stream profiles, the color pixel format
and the depth scale are device/SDK dependent. The defaults below follow the
standard pyorbbecsdk usage for the Gemini 2L; adjust if your unit reports a
different format. The camera intrinsics should ultimately be read from the
device (or from a calibration) rather than hard-coded downstream.
"""
import os

import numpy as np
import cv2

try:
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBFormat, OBStreamType, AlignFilter,
    )
    _HAS_ORBBEC = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_ORBBEC = False

from .paths import SAMPLE_DIR


class OrbbecCapture:
    def __init__(
        self,
        save_directory=str(SAMPLE_DIR),
        rgb_filename="rgb.png",
        depth_filename="depth.npy",
        resolution_width=640,
        resolution_height=480,
        fps=30,
    ):
        """Initialize the capture parameters for the Orbbec Gemini 2L."""
        if not _HAS_ORBBEC:
            raise ImportError(
                "pyorbbecsdk is not installed. Install the Orbbec SDK Python "
                "bindings to use OrbbecCapture (Gemini 2L)."
            )

        self.save_directory = save_directory
        self.rgb_filename = rgb_filename
        self.depth_filename = depth_filename
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.fps = fps

        # Create the output directory if it does not exist.
        os.makedirs(self.save_directory, exist_ok=True)

        # Configure the pipeline and the streaming options.
        self.pipeline = Pipeline()
        self.config = Config()

        # Color stream (request RGB; converted to BGR before saving for OpenCV).
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = color_profiles.get_video_stream_profile(
                self.resolution_width, self.resolution_height, OBFormat.RGB, self.fps
            )
        except Exception:
            # TODO(hardware): fall back to the device default if the exact
            # (resolution, format, fps) combination is not supported.
            color_profile = color_profiles.get_default_video_stream_profile()
        self.config.enable_stream(color_profile)

        # Depth stream (16-bit).
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profiles.get_video_stream_profile(
                self.resolution_width, self.resolution_height, OBFormat.Y16, self.fps
            )
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        self.config.enable_stream(depth_profile)

        # Align the depth frame to the color frame.
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    @staticmethod
    def _color_to_bgr(color_frame):
        """Convert an Orbbec color frame to a BGR uint8 image for OpenCV."""
        h = color_frame.get_height()
        w = color_frame.get_width()
        fmt = color_frame.get_format()
        buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return buf.reshape(h, w, 3)
        if fmt == OBFormat.MJPG:
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        # TODO(hardware): handle other formats (YUYV, etc.) if the device uses them.
        raise ValueError(f"Unsupported Orbbec color format: {fmt}")

    def run_capture(self):
        """Start the Orbbec stream, preview frames, and save on 's' (ESC to quit)."""
        try:
            self.pipeline.start(self.config)
            print("Streaming started. Press 's' to save the image and depth data, or 'ESC' to exit without saving.")
        except Exception as e:
            print(f"Failed to start the Orbbec pipeline: {e}")
            return

        try:
            while True:
                # Wait for a frame set (depth and color).
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                # Align the depth frame to the color frame.
                frames = self.align.process(frames)
                if frames is None:
                    continue
                frames = frames.as_frame_set()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue

                # Convert frames to NumPy arrays.
                color_image = self._color_to_bgr(color_frame)

                dh = depth_frame.get_height()
                dw = depth_frame.get_width()
                # Orbbec depth values are in units of `depth_scale` millimetres.
                depth_scale = depth_frame.get_depth_scale()
                raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(dh, dw)

                # Convert depth to metres.
                depth_image = raw_depth.astype(np.float32) * depth_scale / 1000.0

                # (Optional) colormap of the depth for a clearer preview
                # (this does not alter the saved data).
                max_d = depth_image.max()
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=255 / max_d if max_d > 0 else 1.0),
                    cv2.COLORMAP_JET,
                )

                # Show the color and depth images.
                cv2.imshow('Orbbec Gemini 2L - Color', color_image)
                cv2.imshow('Orbbec Gemini 2L - Depth', depth_colormap)

                # Handle keyboard input.
                key = cv2.waitKey(1) & 0xFF

                # On 's', save and stop.
                if key == ord('s'):
                    rgb_path = os.path.join(self.save_directory, self.rgb_filename)
                    cv2.imwrite(rgb_path, color_image)
                    print(f"Saved RGB image to {rgb_path}")

                    depth_path = os.path.join(self.save_directory, self.depth_filename)
                    np.save(depth_path, depth_image)
                    print(f"Saved depth data to {depth_path}")

                    break  # Exit after saving.

                # On ESC (27), exit without saving.
                elif key == 27:
                    print("Exiting without saving.")
                    break

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Stop the stream and close the windows.
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Stream stopped and windows closed.")


# Run the capture
if __name__ == "__main__":
    capture = OrbbecCapture()
    capture.run_capture()
