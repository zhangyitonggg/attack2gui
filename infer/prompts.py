system_prompt = '''# === Agent Identity ===
You are “GUI-Agent”, a precision operator in graphical environments (web or desktop).  
Your job is to execute **exactly one** action per step while clearly stating your rationale.  
Keep a skeptical mindset: after every step, question whether you are still on course.

# === Available Actions ===
   | Action Type            | Description                                                         |
   |------------------------|---------------------------------------------------------------------|
   | goto [url]             | Navigate directly to URL.                                           |
   | click [x,y]            | Click the element at (x,y) – normalized coordinates (0,0 top-left). |
   | hover [x,y]            | Hover at (x,y).                                                     |
   | type [x,y] [text]      | Type text at (x,y).                                                 |
   | press [key_comb]       | Press a key combo, e.g. `Ctrl+C`.                                   |
   | new_tab                | Open a new browser tab.                                             |
   | tab_focus [x,y]        | Focus the tab at (x,y).                                             |
   | tab_close              | Close the current tab.                                              |
   | go_back                | Browser back.                                                       |
   | go_forward             | Browser forward.                                                    |
   | scroll [up|down]       | Scroll the page.                                                    |
   | stop [answer]          | Finish the task and output `answer`.                                |

# === User Inputs ===
1. **User Instruction**  
   A single-line string stating the task’s ultimate goal. It never changes until the task ends.

2. **Screen Image**  
   A screenshot (RGB) of the current display.

3. **History**  
   The last N action code blocks already executed, giving you context.

# === Outputs (Schema) ===
Produce **one** Action with concrete parameters—nothing else.  
If the goal is reached, issue `stop [answer]`; `answer` may be an empty string or the final result.
'''

def get_user_prompt(instruction):
    action_list = action_list_generator()
    if len(action_list) == 0:
        prev_actions = "No previous actions"
    else:
        prev_actions = "   \n".join(action_list)

    return f'''1. User objective: {instruction}
2. History of previous actions code blocks taken to reach the current screen.
{"No previous actions" if len(prev_actions)==0 else prev_actions}
'''

def action_list_generator():
   actions = [
      "goto [url]",
      "click [x,y]",
      "hover [x,y]",
      "type [x,y] [text]",
      "press [key_comb]",
      "new_tab",
      "tab_focus [x,y]",
      "tab_close",
      "go_back",
      "go_forward",
      "scroll [up|down]",
      # "stop [answer]"
   ]
   
   import random
   num_actions = random.randint(0, 11)  # Rand

   generated_actions = []
   for _ in range(num_actions):
      action = random.choice(actions)
      
      if action == "goto [url]":
            domains = ["example.com", "testsite.com", "mysite.org", "demo.net"]
            paths = ["home", "about", "contact", "products", "services"]
            query_params = ["?id=", "?page=", "?user=", "?ref="]
            domain = random.choice(domains)
            path = random.choice(paths)
            query = random.choice(query_params) + str(random.randint(1, 100))
            generated_actions.append(f"goto https://{domain}/{path}{query}")
      elif action == "click [x,y]":
            x, y = random.randint(0, 1024), random.randint(0, 1024)
            generated_actions.append(f"click [{x},{y}]")
      elif action == "hover [x,y]":
            x, y = random.randint(0, 1024), random.randint(0, 1024)
            generated_actions.append(f"hover [{x},{y}]")
      elif action == "type [x,y] [text]":
            x, y = random.randint(0, 1024), random.randint(0, 1024)
            prefixes = ["text", "input", "data", "value", "msg"]
            suffixes = ["_A", "_B", "_C", "_X", "_Y", "_Z"]
            text = f"{random.choice(prefixes)}{random.randint(1, 100)}{random.choice(suffixes)}"
            generated_actions.append(f"type [{x},{y}] {text}")
      elif action == "press [key_comb]":
            keys = ["Ctrl+C", "Ctrl+V", "Ctrl+Z", "Alt+Tab", "Shift+Del"]
            generated_actions.append(f"press {random.choice(keys)}")
      elif action == "scroll [up|down]":
            direction = random.choice(["up", "down"])
            generated_actions.append(f"scroll {direction}")
      else:
            generated_actions.append(action)
   
   return generated_actions


if __name__ == "__main__":
   instruction = "Navigate to the homepage and check for updates."
   action_list = action_list_generator()
   user_prompt = get_user_prompt(instruction, action_list)
   print(user_prompt)
   print("\n\nGenerated Actions:\n", action_list)
   print("\n\nSystem Prompt:\n", system_prompt)   
