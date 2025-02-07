# the functionality of this code is: 
# ✅ Sends alerts when a person enters a restricted zone
# ✅ Can be extended for email, SMS, or webhook notifications

import smtplib
import requests

# Example restricted areas (latitude, longitude)
RESTRICTED_ZONES = [
    (50.2, 49.8),  # Replace with actual restricted locations
    (51.0, 50.5),
]

def is_in_restricted_zone(x, y):
    """ Check if a person is inside any restricted area """
    for zone_x, zone_y in RESTRICTED_ZONES:
        if abs(x - zone_x) < 0.01 and abs(y - zone_y) < 0.01:  # Small tolerance
            return True
    return False

def send_email_alert(message):
    """ Send an email alert """
    sender_email = "your_email@example.com"
    receiver_email = "alert_receiver@example.com"
    password = "your_email_password"

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def trigger_alert(x, y):
    """ Trigger alert if person enters restricted area """
    if is_in_restricted_zone(x, y):
        alert_msg = f"⚠️ ALERT: Unauthorized person detected at ({x}, {y})"
        print(alert_msg)
        send_email_alert(alert_msg)  # Send an email
        requests.post("https://your-webhook-url.com", json={"alert": alert_msg})  # Webhook alert
