def tokenize_txt(tokenizer, max_text_length, txt):
    '''
    Tokenizes the text by trimming the appropriate txt
    :param tokenizer:
    param config:
    :param txt:
    '''

    txt_input_ids = tokenizer.encode(txt, add_special_tokens=False) + [tokenizer.sep_token_id]

    # Add 1 to account for <start> token
    tot_length = len(txt_input_ids) + 1

    # Don't need to trim text
    if tot_length <= max_text_length:
        trunc_input_ids = [tokenizer.pad_token_id] * max_text_length
        trunc_input_ids[:tot_length] = txt_input_ids

    # Trim text
    else:
        num_trim = tot_length - max_text_length
        new_txt_input_ids = txt_input_ids[:-num_trim]
        trunc_input_ids = new_txt_input_ids
    
    trunc_input_ids = [tokenizer.cls_token_id] + trunc_input_ids
    
    return trunc_input_ids

def tokenize_concat_txt(tokenizer, max_text_length, txt, explanations):
    '''
    Tokenizes the text by trimming the explanations txt
    :param tokenizer:
    param config:
    :param txt:
    '''

    txt_input_ids = tokenizer.encode(explanations, add_special_tokens=False) + [tokenizer.sep_token_id] + tokenizer.encode(txt, add_special_tokens=False) + [tokenizer.sep_token_id]

    # Add 1 to account for <start> token
    tot_length = len(txt_input_ids) + 1

    # Don't need to trim text
    if tot_length <= max_text_length:
        trunc_input_ids = [tokenizer.pad_token_id] * max_text_length
        trunc_input_ids[:tot_length] = txt_input_ids

    # Trim text
    else:
        num_trim = tot_length - max_text_length
        txt_input_ids = tokenizer.encode(explanations, add_special_tokens=False)[:-num_trim] 
        new_txt_input_ids = txt_input_ids + [tokenizer.sep_token_id] + tokenizer.encode(txt, add_special_tokens=False) + [tokenizer.sep_token_id]
        trunc_input_ids = new_txt_input_ids
    
    trunc_input_ids = [tokenizer.cls_token_id] + trunc_input_ids
    
    return trunc_input_ids