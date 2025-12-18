#!/usr/bin/env python3
"""
Merge video recording with voiceover audio.
Usage: python scripts/merge_video_audio.py <video_file>
"""

import subprocess
import sys
import os

def merge(video_path, audio_path="demo_audio/FULL_VOICEOVER.mp3", output_path=None):
    """Merge video with audio using ffmpeg."""
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_with_voiceover.mp4"
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║                    MERGING VIDEO + AUDIO                       ║
╠════════════════════════════════════════════════════════════════╣
║  Video:  {video_path:<50} ║
║  Audio:  {audio_path:<50} ║
║  Output: {output_path:<50} ║
╚════════════════════════════════════════════════════════════════╝
""")
    
    # FFmpeg command to merge video and audio
    # -i: input files
    # -c:v copy: copy video codec (no re-encoding)
    # -map 0:v: use video from first input
    # -map 1:a: use audio from second input
    # -shortest: end when shortest input ends
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,  # Input video
        "-i", audio_path,  # Input audio
        "-c:v", "copy",  # Copy video (no re-encode)
        "-map", "0:v:0",  # Video from first input
        "-map", "1:a:0",  # Audio from second input
        "-shortest",  # End at shortest stream
        output_path
    ]
    
    print("Processing...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"""
✅ SUCCESS!

   Output: {output_path}
   Size:   {size_mb:.1f} MB

   Ready to upload to LinkedIn/YouTube!
""")
    else:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
Usage: python scripts/merge_video_audio.py <video_file>

Examples:
  python scripts/merge_video_audio.py recording.mp4
  python scripts/merge_video_audio.py ~/Videos/demo.mkv
  
The script will:
1. Take your screen recording (video)
2. Add demo_audio/FULL_VOICEOVER.mp3 (audio)
3. Output: <video_file>_with_voiceover.mp4
""")
        sys.exit(1)
    
    video_file = sys.argv[1]
    merge(video_file)
