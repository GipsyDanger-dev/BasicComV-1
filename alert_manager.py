import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging


class AlertManager:
    def __init__(self, config):
        self.config = config
        self.alert_history = []
        self.logger = logging.getLogger(__name__)
        self.last_alert_time = {}
        self.cooldown_seconds = 10
    
    def should_send_alert(self, alert_type):
        now = datetime.now()
        if alert_type in self.last_alert_time:
            elapsed = (now - self.last_alert_time[alert_type]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        
        self.last_alert_time[alert_type] = now
        return True
    
    def check_thresholds(self, stats):
        alerts = []
        
        if not self.config['alerts'].get('enabled', False):
            return alerts
        
        thresholds = self.config['alerts'].get('thresholds', {})
        total_objects = sum(stats.class_counts.values()) if hasattr(stats, 'class_counts') else 0
        
        max_objects = thresholds.get('max_objects', float('inf'))
        min_objects = thresholds.get('min_objects', 0)
        
        if total_objects > max_objects:
            if self.should_send_alert('max_objects'):
                alerts.append({
                    'type': 'threshold',
                    'message': f'Too many objects detected: {total_objects} (max: {max_objects})',
                    'severity': 'warning',
                    'timestamp': datetime.now()
                })
        
        if total_objects < min_objects and total_objects > 0:
            if self.should_send_alert('min_objects'):
                alerts.append({
                    'type': 'threshold',
                    'message': f'Too few objects detected: {total_objects} (min: {min_objects})',
                    'severity': 'info',
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def send_email(self, subject, body):
        if not self.config['alerts']['email'].get('enabled', False):
            return False
        
        try:
            email_config = self.config['alerts']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email sent: {subject}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def send_webhook(self, data):
        if not self.config['alerts']['webhook'].get('enabled', False):
            return False
        
        try:
            webhook_url = self.config['alerts']['webhook']['url']
            response = requests.post(webhook_url, json=data, timeout=5)
            
            if response.status_code == 200:
                self.logger.info(f"Webhook sent successfully")
                return True
            else:
                self.logger.warning(f"Webhook failed: {response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {e}")
            return False
    
    def process_alerts(self, alerts):
        for alert in alerts:
            self.alert_history.append(alert)
            
            subject = f"[CompV Alert] {alert['severity'].upper()}: {alert['type']}"
            body = f"""
CompV Object Detection Alert

Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert['message']}

---
This is an automated alert from CompV Object Detection System.
"""
            
            self.send_email(subject, body)
            
            webhook_data = {
                'alert_type': alert['type'],
                'severity': alert['severity'],
                'message': alert['message'],
                'timestamp': alert['timestamp'].isoformat()
            }
            self.send_webhook(webhook_data)
    
    def get_alert_history(self, limit=10):
        return self.alert_history[-limit:]
