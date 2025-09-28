# Technical Implementation Guide - Vision LLM Bubble Detection

## Overview

This guide provides specific technical implementation details and code examples for each phase of the Vision LLM-based bubble detection system.

---

## Phase 1: Infrastructure & Data Preparation

### 1.1 Frame Extraction Pipeline

```python
# extract_bubble_analysis_frames.py
import json
import os
import cv2
from datetime import datetime, timedelta
from pathlib import Path

class BubbleFrameExtractor:
    def __init__(self, test_periods_file="analysis/cam8_refined_test_periods_precise.json"):
        with open(test_periods_file, 'r') as f:
            self.test_periods = json.load(f)
        self.output_base = "analysis/bubble_frames"

    def extract_frames(self, normal_interval=2.0, critical_interval=0.5):
        """Extract frames at specified intervals"""
        for test_idx, test_period in enumerate(self.test_periods, 1):
            print(f"Processing Test Period {test_idx}")

            # Extract frames from cam-1 through cam-7
            for cam_num in range(1, 8):
                cam_name = f"cam-{cam_num}"
                self.extract_camera_frames(
                    test_period, test_idx, cam_name,
                    normal_interval, critical_interval
                )

    def extract_camera_frames(self, test_period, test_idx, cam_name, normal_interval, critical_interval):
        """Extract frames for a specific camera during test period"""
        # Implementation details for frame extraction
        # Check for video files in date folders
        # Extract frames at specified intervals
        # Save with meaningful names
        pass

# Usage
extractor = BubbleFrameExtractor()
extractor.extract_frames()
```

### 1.2 Frame Quality Assessment

```python
# assess_frame_quality.py
import cv2
import numpy as np
from pathlib import Path
import json

class FrameQualityAssessor:
    def __init__(self, frames_dir="analysis/bubble_frames"):
        self.frames_dir = Path(frames_dir)
        self.quality_metrics = {}

    def assess_quality(self, image_path):
        """Assess frame quality based on multiple metrics"""
        img = cv2.imread(str(image_path))

        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness
        brightness = np.mean(gray)

        # Contrast
        contrast = np.std(gray)

        return {
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'overall_score': self.calculate_overall_score(sharpness, brightness, contrast)
        }

    def calculate_overall_score(self, sharpness, brightness, contrast):
        """Calculate overall quality score"""
        # Normalize and weight metrics
        score = (
            min(sharpness / 100, 1.0) * 0.5 +  # Sharpness weight: 50%
            (1 - abs(brightness - 128) / 128) * 0.3 +  # Optimal brightness weight: 30%
            min(contrast / 50, 1.0) * 0.2  # Contrast weight: 20%
        )
        return score

    def find_best_frames(self, top_n=5):
        """Find best quality frames for each test/camera combination"""
        results = {}

        for test_dir in self.frames_dir.glob("test_*"):
            test_name = test_dir.name
            results[test_name] = {}

            for cam_dir in test_dir.glob("cam_*"):
                cam_name = cam_dir.name
                frames = list(cam_dir.glob("*.jpg"))

                # Assess all frames
                frame_scores = []
                for frame in frames:
                    metrics = self.assess_quality(frame)
                    frame_scores.append({
                        'path': str(frame),
                        'metrics': metrics
                    })

                # Sort by overall score and get top N
                frame_scores.sort(key=lambda x: x['metrics']['overall_score'], reverse=True)
                results[test_name][cam_name] = frame_scores[:top_n]

        # Save results
        with open('best_frames_for_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

# Usage
assessor = FrameQualityAssessor()
best_frames = assessor.find_best_frames()
```

### 1.3 API Setup & Cost Estimation

```python
# config/api_config.py
import os
from dotenv import load_dotenv

load_dotenv()

class APIConfig:
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    RATE_LIMIT_PER_MINUTE = 50

    # Pricing (as of Oct 2024)
    VISION_COST_PER_IMAGE = 0.003  # $3 per 1K images
    TEXT_COST_PER_1K_TOKENS = 0.003

    # Image settings
    MAX_IMAGE_SIZE = 1024 * 1024 * 5  # 5MB
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp']

# estimate_analysis_cost.py
import json
from pathlib import Path
from config.api_config import APIConfig

def estimate_costs(frames_dir="analysis/bubble_frames"):
    """Estimate API costs for analyzing all frames"""
    total_frames = 0

    frames_path = Path(frames_dir)
    for test_dir in frames_path.glob("test_*"):
        for cam_dir in test_dir.glob("cam_*"):
            total_frames += len(list(cam_dir.glob("*.jpg")))

    image_cost = total_frames * APIConfig.VISION_COST_PER_IMAGE
    # Estimate ~100 tokens per request/response
    text_cost = (total_frames * 200 / 1000) * APIConfig.TEXT_COST_PER_1K_TOKENS

    total_cost = image_cost + text_cost

    print(f"Frame Analysis Cost Estimation:")
    print(f"Total frames: {total_frames}")
    print(f"Image analysis cost: ${image_cost:.2f}")
    print(f"Text token cost: ${text_cost:.2f}")
    print(f"Total estimated cost: ${total_cost:.2f}")

    return {
        'total_frames': total_frames,
        'image_cost': image_cost,
        'text_cost': text_cost,
        'total_cost': total_cost
    }
```

---

## Phase 2: Vision LLM Integration & Prompt Engineering

### 2.1 Basic Bubble Detection

```python
# vision_llm/bubble_analyzer.py
import base64
import json
from pathlib import Path
from anthropic import Anthropic
from typing import Dict, Any
import time
from config.api_config import APIConfig

class BubbleAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=APIConfig.ANTHROPIC_API_KEY)
        self.model = "claude-3-5-sonnet-20241022"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def analyze_frame(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single frame for bubbles"""
        image_data = self.encode_image(image_path)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": self.get_analysis_prompt()
                    }
                ]
            }]
        )

        # Parse response
        return self.parse_response(response.content[0].text, image_path)

    def get_analysis_prompt(self) -> str:
        """Get the bubble analysis prompt"""
        return """Analyze this underwater image of a high-pressure gas meter for air bubbles that indicate leaks.

Please provide a structured analysis:

1. BUBBLE DETECTION:
   - Are there visible air bubbles? (yes/no)
   - Bubble count estimate: (none/few: 1-5/moderate: 6-20/many: 20+)

2. BUBBLE CHARACTERISTICS:
   - Size distribution: (tiny <2mm / small 2-5mm / medium 5-10mm / large >10mm)
   - Movement pattern: (rising/stationary/turbulent)
   - Origin point: (describe location relative to meter components)

3. LEAK ASSESSMENT:
   - Leak severity: (no leak/minor/moderate/severe)
   - Confidence level: (low/medium/high)
   - Reasoning: (brief explanation)

4. ADDITIONAL OBSERVATIONS:
   - Meter visibility: (clear/partially obscured/poor)
   - Water conditions: (clear/murky/debris present)
   - Other anomalies: (describe any)

Format your response as JSON with these exact keys:
{
  "bubbles_detected": boolean,
  "bubble_count": "none|few|moderate|many",
  "bubble_sizes": ["tiny", "small", "medium", "large"],
  "movement_pattern": "rising|stationary|turbulent|mixed",
  "origin_location": "description",
  "leak_severity": "no_leak|minor|moderate|severe",
  "confidence": "low|medium|high",
  "reasoning": "explanation",
  "meter_visibility": "clear|partial|poor",
  "water_conditions": "description",
  "anomalies": "description or null"
}"""

    def parse_response(self, response_text: str, image_path: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]

            result = json.loads(json_str)
            result['image_path'] = image_path
            result['timestamp'] = time.time()

            return result
        except Exception as e:
            return {
                'error': str(e),
                'raw_response': response_text,
                'image_path': image_path,
                'timestamp': time.time()
            }

# vision_llm/prompts.py
class PromptTemplates:
    """Collection of optimized prompts for different scenarios"""

    BASIC_DETECTION = """[Basic prompt as shown above]"""

    DETAILED_ANALYSIS = """[More detailed version with specific guidance]"""

    RAPID_SCREENING = """Quick assessment for bubble presence:
    Look for air bubbles in this underwater meter image.
    Response format: {"leak_likely": true/false, "confidence": "low/medium/high"}"""
```

### 2.2 Temporal Analysis

```python
# vision_llm/temporal_analyzer.py
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

class TemporalAnalyzer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.detection_history = defaultdict(list)

    def add_detection(self, camera: str, timestamp: float, detection: Dict[str, Any]):
        """Add detection result to history"""
        self.detection_history[camera].append({
            'timestamp': timestamp,
            'detection': detection
        })

    def analyze_patterns(self, camera: str) -> Dict[str, Any]:
        """Analyze temporal patterns for a camera"""
        if camera not in self.detection_history:
            return {'error': 'No data for camera'}

        detections = self.detection_history[camera]

        # Sort by timestamp
        detections.sort(key=lambda x: x['timestamp'])

        # Analyze patterns
        bubble_timeline = []
        severity_timeline = []

        for det in detections:
            bubble_timeline.append(1 if det['detection'].get('bubbles_detected', False) else 0)
            severity_map = {'no_leak': 0, 'minor': 1, 'moderate': 2, 'severe': 3}
            severity_timeline.append(severity_map.get(det['detection'].get('leak_severity', 'no_leak'), 0))

        # Calculate trends
        patterns = {
            'total_detections': len(detections),
            'bubble_frequency': sum(bubble_timeline) / len(bubble_timeline) if bubble_timeline else 0,
            'avg_severity': np.mean(severity_timeline) if severity_timeline else 0,
            'increasing_trend': self.detect_trend(severity_timeline),
            'burst_events': self.detect_bursts(bubble_timeline),
            'persistent_leak': self.detect_persistent(bubble_timeline)
        }

        return patterns

    def detect_trend(self, timeline: List[float]) -> str:
        """Detect if severity is increasing, decreasing, or stable"""
        if len(timeline) < 3:
            return 'insufficient_data'

        # Simple linear regression
        x = np.arange(len(timeline))
        slope = np.polyfit(x, timeline, 1)[0]

        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

    def detect_bursts(self, timeline: List[int], threshold=0.7) -> List[Dict]:
        """Detect burst events (sudden increases in bubble activity)"""
        bursts = []

        for i in range(len(timeline) - self.window_size):
            window = timeline[i:i + self.window_size]
            if sum(window) / len(window) > threshold:
                bursts.append({
                    'start_idx': i,
                    'end_idx': i + self.window_size,
                    'intensity': sum(window) / len(window)
                })

        return bursts

    def detect_persistent(self, timeline: List[int], min_duration=10) -> bool:
        """Detect persistent leaks (continuous bubble detection)"""
        consecutive = 0
        max_consecutive = 0

        for detection in timeline:
            if detection:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        return max_consecutive >= min_duration
```

---

## Phase 3: Analysis & Validation

### 3.1 Batch Processing

```python
# run_full_analysis.py
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from vision_llm.bubble_analyzer import BubbleAnalyzer
from vision_llm.temporal_analyzer import TemporalAnalyzer
import pickle

class BatchAnalyzer:
    def __init__(self, max_workers=5):
        self.analyzer = BubbleAnalyzer()
        self.temporal = TemporalAnalyzer()
        self.max_workers = max_workers
        self.checkpoint_file = "analysis_checkpoint.pkl"

    def run_analysis(self, frames_dir="analysis/bubble_frames"):
        """Run complete analysis on all frames"""
        frames_path = Path(frames_dir)
        all_frames = []

        # Collect all frames
        for test_dir in frames_path.glob("test_*"):
            for cam_dir in test_dir.glob("cam_*"):
                for frame in cam_dir.glob("*.jpg"):
                    all_frames.append({
                        'path': str(frame),
                        'test': test_dir.name,
                        'camera': cam_dir.name,
                        'timestamp': self.extract_timestamp(frame.name)
                    })

        # Load checkpoint if exists
        completed = self.load_checkpoint()
        remaining = [f for f in all_frames if f['path'] not in completed]

        print(f"Total frames: {len(all_frames)}")
        print(f"Already completed: {len(completed)}")
        print(f"Remaining: {len(remaining)}")

        # Process in batches with rate limiting
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for frame_info in remaining:
                future = executor.submit(self.analyze_frame_safe, frame_info)
                futures.append(future)

                # Rate limiting
                if len(futures) >= 50:  # Process in chunks
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            results.append(result)
                            self.save_checkpoint(result['path'])
                    futures = []
                    time.sleep(60)  # Rate limit pause

            # Process remaining
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                    self.save_checkpoint(result['path'])

        # Save final results
        self.save_results(results)
        return results

    def analyze_frame_safe(self, frame_info):
        """Analyze frame with error handling"""
        try:
            result = self.analyzer.analyze_frame(frame_info['path'])
            result.update(frame_info)

            # Add to temporal analysis
            self.temporal.add_detection(
                frame_info['camera'],
                frame_info['timestamp'],
                result
            )

            return result
        except Exception as e:
            print(f"Error analyzing {frame_info['path']}: {e}")
            return None

    def extract_timestamp(self, filename):
        """Extract timestamp from filename"""
        # Implementation depends on naming convention
        parts = filename.split('_')
        return float(parts[-1].replace('.jpg', ''))

    def save_results(self, results):
        """Save analysis results"""
        output_dir = Path("analysis/vision_llm_results")
        output_dir.mkdir(exist_ok=True)

        # Save raw results
        with open(output_dir / "all_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Generate summaries by test period
        summaries = self.generate_summaries(results)
        with open(output_dir / "test_summaries.json", 'w') as f:
            json.dump(summaries, f, indent=2)

    def generate_summaries(self, results):
        """Generate summaries by test period"""
        summaries = defaultdict(lambda: {
            'total_frames': 0,
            'bubbles_detected': 0,
            'severity_counts': defaultdict(int),
            'cameras_affected': set()
        })

        for result in results:
            test = result['test']
            summary = summaries[test]

            summary['total_frames'] += 1
            if result.get('bubbles_detected'):
                summary['bubbles_detected'] += 1
                summary['cameras_affected'].add(result['camera'])

            severity = result.get('leak_severity', 'no_leak')
            summary['severity_counts'][severity] += 1

        # Convert sets to lists for JSON serialization
        for test in summaries:
            summaries[test]['cameras_affected'] = list(summaries[test]['cameras_affected'])

        return dict(summaries)

    def load_checkpoint(self):
        """Load checkpoint of completed analyses"""
        if Path(self.checkpoint_file).exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return set()

    def save_checkpoint(self, completed_path):
        """Save checkpoint"""
        completed = self.load_checkpoint()
        completed.add(completed_path)
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(completed, f)

# Main execution
if __name__ == "__main__":
    analyzer = BatchAnalyzer()
    results = analyzer.run_analysis()
    print(f"Analysis complete. Processed {len(results)} frames.")
```

---

## Phase 4: Results & Reporting

### 4.1 Report Generation

```python
# generate_technical_report.py
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class ReportGenerator:
    def __init__(self, results_dir="analysis/vision_llm_results"):
        self.results_dir = Path(results_dir)
        self.load_results()

    def load_results(self):
        """Load analysis results"""
        with open(self.results_dir / "all_results.json", 'r') as f:
            self.results = json.load(f)

        with open(self.results_dir / "test_summaries.json", 'r') as f:
            self.summaries = json.load(f)

    def generate_report(self):
        """Generate comprehensive technical report"""
        report = []

        # Executive Summary
        report.append("# Bubble Detection System - Technical Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Executive Summary\n")

        total_frames = sum(s['total_frames'] for s in self.summaries.values())
        total_detections = sum(s['bubbles_detected'] for s in self.summaries.values())
        detection_rate = (total_detections / total_frames * 100) if total_frames > 0 else 0

        report.append(f"- Total frames analyzed: {total_frames}")
        report.append(f"- Bubble detections: {total_detections} ({detection_rate:.1f}%)")
        report.append(f"- Test periods analyzed: {len(self.summaries)}")

        # Test-by-test breakdown
        report.append("\n## Test Period Analysis\n")

        for test, summary in self.summaries.items():
            report.append(f"\n### {test}")
            report.append(f"- Frames analyzed: {summary['total_frames']}")
            report.append(f"- Bubbles detected: {summary['bubbles_detected']}")
            report.append(f"- Cameras affected: {', '.join(summary['cameras_affected']) if summary['cameras_affected'] else 'None'}")

            # Severity breakdown
            report.append("\nSeverity Distribution:")
            for severity, count in summary['severity_counts'].items():
                percentage = (count / summary['total_frames'] * 100)
                report.append(f"- {severity}: {count} ({percentage:.1f}%)")

        # Save report
        report_path = Path("reports/bubble_detection_technical_report.md")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        # Generate visualizations
        self.generate_visualizations()

        return report_path

    def generate_visualizations(self):
        """Generate charts and graphs"""
        # Detection rate by test period
        plt.figure(figsize=(10, 6))
        tests = list(self.summaries.keys())
        detection_rates = [
            (s['bubbles_detected'] / s['total_frames'] * 100) if s['total_frames'] > 0 else 0
            for s in self.summaries.values()
        ]

        plt.bar(tests, detection_rates)
        plt.xlabel('Test Period')
        plt.ylabel('Detection Rate (%)')
        plt.title('Bubble Detection Rate by Test Period')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/detection_rates.png')
        plt.close()

        # Severity distribution
        severity_totals = defaultdict(int)
        for summary in self.summaries.values():
            for severity, count in summary['severity_counts'].items():
                severity_totals[severity] += count

        plt.figure(figsize=(8, 8))
        plt.pie(severity_totals.values(), labels=severity_totals.keys(), autopct='%1.1f%%')
        plt.title('Overall Leak Severity Distribution')
        plt.savefig('reports/severity_distribution.png')
        plt.close()
```

### 4.2 Interactive Demo

```python
# demo/bubble_detection_demo.py
from flask import Flask, render_template, request, jsonify
import base64
from vision_llm.bubble_analyzer import BubbleAnalyzer
import os

app = Flask(__name__)
analyzer = BubbleAnalyzer()

@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    try:
        # Get image from request
        image_data = request.json['image']

        # Save temporarily
        temp_path = 'temp_frame.jpg'
        with open(temp_path, 'wb') as f:
            f.write(base64.b64decode(image_data.split(',')[1]))

        # Analyze
        result = analyzer.analyze_frame(temp_path)

        # Clean up
        os.remove(temp_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/examples')
def examples():
    """Get example frames"""
    examples = []
    example_dir = Path('demo/example_frames')

    for img in example_dir.glob('*.jpg'):
        with open(img, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
            examples.append({
                'name': img.name,
                'data': f'data:image/jpeg;base64,{img_data}'
            })

    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## Optimization Tips

### 1. Cost Reduction

- Resize images to 1024x768 before sending
- Use JPEG compression (quality=85)
- Cache all API responses
- Batch similar frames

### 2. Speed Improvements

- Parallel processing with rate limiting
- Pre-filter obvious non-leak frames
- Progressive analysis (start with samples)

### 3. Accuracy Enhancement

- Validate on known test cases
- Iterative prompt refinement
- Cross-reference with control panel data
- Manual review of edge cases
