# Object detection for pi camera

## Installation
- Open terminal  
- Navigate to dwonloaded folder  
- Run pip3 install -r requirements.txt  
- Make sure camera interface is enabled in Raspberry Pi Configuration  
- Reboot  

## Detect objects
- Open terminal  
- Navigate to downloaded folder  
- Run python3  
- Type "from object_detection import show_image" and press enter  
- Type "show_image()" and press enter  

## Basic example
```python
"""Prints "Person detected" every time a person is detected for 60 seconds"""

import time
from object_detection import *

add_listener("person", lambda: print("Person detected"))
start_detection()
time.sleep(60)
stop_detection()
```
  
