import os
def apply_RL_prompt(chunk, args, budget):
    if args.prompt_type == "deepseek3" and os.environ['tip'] == "Ahead":
        return RL_deepseek3_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "remaining":
        return RL_deepseek3_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "prompt-based":
        return deepseek3_prompt_based(chunk, budget)
    elif os.environ['tip'] == "default":
        return chunk

    
    

def RL_deepseek3_prompt(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"\n<remaining>[{budget} token]</remaining>\n"
        chunk[i] = head + find_strings + tail
    return chunk
        
    
def deepseek3_prompt_based(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"You should finish thinking with in {budget} tokens.\n"
        chunk[i] = head + find_strings + tail
    return chunk