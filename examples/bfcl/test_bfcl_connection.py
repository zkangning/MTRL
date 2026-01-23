import requests
import json
import sys
import argparse

def test_fetch_env_profile(base_url, env_type="bfcl", split="train"):
    endpoint = f"{base_url.rstrip('/')}/get_env_profile"
    
    print(f"\n{'='*10} BFCL Connection Test {'='*10}")
    print(f"Target URL: {endpoint}")
    
    payload = {
        "env_type": env_type,
        "params": {"split": split}
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # 发送请求
        resp = requests.post(endpoint, json=payload, timeout=30)
        
        # 打印状态码
        print(f"\nResponse Status Code: {resp.status_code}")
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                
                # 尝试解析 Task List
                task_list = []
                if isinstance(data, dict):
                    # 检查常见的字段名
                    if "data" in data:
                        task_list = data["data"]
                        print("Found tasks in field: 'data'")
                    elif "task_ids" in data:
                        task_list = data["task_ids"]
                        print("Found tasks in field: 'task_ids'")
                    else:
                        print(f"Warning: Unknown dict structure. Keys: {list(data.keys())}")
                        # 如果没找到标准字段，打印整个字典的前部分供调试
                        print(f"Raw Response (Head): {str(data)[:200]}...")
                elif isinstance(data, list):
                    task_list = data
                    print("Response is a direct List")
                
                if task_list:
                    print(f"\n✅ SUCCESS: Retrieved {len(task_list)} tasks.")
                    print(f"First 5 task IDs: {task_list[:5]}")
                    print(f"Last task ID: {task_list[-1]}")
                else:
                    print("\n⚠️ WARNING: Request succeeded (200 OK) but the task list is empty.")
                    print(f"Full Response: {data}")
                    
            except json.JSONDecodeError:
                print("\n❌ ERROR: Failed to decode JSON response.")
                print(f"Raw Text: {resp.text}")
        else:
            print(f"\n❌ FAILED: Server returned error status.")
            print(f"Response Text: {resp.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ CONNECTION ERROR: Could not connect to {base_url}")
        print("Please ensure the BFCL EnvService (Java/Python server) is running.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BFCL EnvService Connection")
    parser.add_argument("--base_url", type=str, default="http://localhost:8801", help="EnvService URL")
    parser.add_argument("--env_type", type=str, default="bfcl", help="Environment Type")
    parser.add_argument("--split", type=str, default="val", help="Dataset split")
    
    args = parser.parse_args()
    
    test_fetch_env_profile(args.base_url, args.env_type, args.split)
