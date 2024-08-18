def generate_chat(input_text, output_text=None, prefix_chat=None):
    chat = [
        {"role": "user", "content": input_text},
    ]
    if output_text is not None:
        chat.append({"role": "assistant", "content": output_text})
    if prefix_chat is not None:
        chat = prefix_chat + chat
    return chat

def generate_chat_v2(conversation_list, output_text=None, prefix_chat=None):
    # if conversation_list is a string, pass it to generate_chat
    if isinstance(conversation_list, str):
        return generate_chat(conversation_list, output_text, prefix_chat)
    
    # else from conversation_list, create a chat list alternating between user (even positions) and assistant (odd positions)
    
    chat = []
    for i, content in enumerate(conversation_list):
        if i % 2 == 0:
            chat.append({"role": "user", "content": content})
        else:
            chat.append({"role": "assistant", "content": content})

    if prefix_chat is not None:
        chat = prefix_chat + chat
    return chat