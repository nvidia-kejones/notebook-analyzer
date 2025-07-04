#!/usr/bin/env python3
"""
Test script to verify streaming functionality
"""
import requests
import time
import json

def test_streaming():
    try:
        # Test health endpoint first
        print("Testing health endpoint...")
        response = requests.get('http://localhost:8080/health', timeout=5)
        print(f'Health check: {response.status_code}')
        
        if response.status_code != 200:
            print("Service not healthy, aborting test")
            return
        
        # Test streaming endpoint
        print('\nTesting streaming endpoint...')
        print('URL: https://github.com/nvidia-kejones/launchables/blob/main/hugging-face-intro.ipynb')
        
        start_time = time.time()
        response = requests.post('http://localhost:8080/analyze-stream', 
                               files={'url': (None, 'https://github.com/nvidia-kejones/launchables/blob/main/hugging-face-intro.ipynb')},
                               stream=True, timeout=120)
        
        print(f'Stream response status: {response.status_code}')
        print(f'Content-Type: {response.headers.get("Content-Type", "Not set")}')
        
        if response.status_code == 200:
            print('\n=== STREAMING DATA ===')
            message_count = 0
            progress_messages = []
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        message_count += 1
                        elapsed = time.time() - start_time
                        
                        print(f'[{elapsed:.1f}s] {data.get("type", "unknown")}: {data.get("message", "")[:100]}...')
                        
                        if data.get('type') == 'progress':
                            progress_messages.append((elapsed, data.get('message', '')))
                        elif data.get('type') == 'complete':
                            print(f'[{elapsed:.1f}s] ANALYSIS COMPLETE!')
                            break
                        elif data.get('type') == 'error':
                            print(f'[{elapsed:.1f}s] ERROR: {data.get("message", "")}')
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f'JSON decode error: {e}')
                        print(f'Raw line: {line}')
            
            print(f'\n=== SUMMARY ===')
            print(f'Total messages: {message_count}')
            print(f'Progress messages: {len(progress_messages)}')
            print(f'Total time: {time.time() - start_time:.1f}s')
            
            if progress_messages:
                print('\nProgress timeline:')
                for elapsed, msg in progress_messages[:10]:  # Show first 10
                    print(f'  {elapsed:.1f}s: {msg}')
                if len(progress_messages) > 10:
                    print(f'  ... and {len(progress_messages) - 10} more messages')
            
            # Check if we got real-time streaming
            if len(progress_messages) > 1:
                first_msg_time = progress_messages[0][0]
                last_msg_time = progress_messages[-1][0]
                if last_msg_time - first_msg_time > 1.0:
                    print('✅ REAL-TIME STREAMING WORKING!')
                else:
                    print('❌ Messages came all at once (not real-time)')
            else:
                print('❌ No progress messages received')
                
        else:
            print(f'Error: {response.text[:500]}')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_streaming() 