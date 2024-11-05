from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json
from jinja2 import Environment, PackageLoader, select_autoescape

from vts.models import AnalysisReport, Topic, Segment

class ReportGenerator:
    def __init__(self):
        self.env = Environment(
            loader=PackageLoader('vts', 'templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
    def generate_markdown(self, report: AnalysisReport) -> str:
        template = self.env.get_template('report.md')
        return template.render(report=report)
    
    def save_report(self, report: AnalysisReport, output_dir: Path) -> Path:
        markdown = self.generate_markdown(report)
        output_path = output_dir / 'analysis_report.md'
        output_path.write_text(markdown)
        return output_path