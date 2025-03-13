import torch
def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

import torch

def run_florence2(task_prompt, text_input=None, model=None, processor=None, image=None):
    """
    Run the Florence-2 model for vision-language tasks.
    
    Args:
        task_prompt (str): The task-specific prompt (e.g., "<CAPTION>").
        text_input (str, optional): Additional text input to append to the task prompt.
        model: Initialized Florence-2 model.
        processor: Florence-2 processor for input preprocessing.
        image: Input image (e.g., PIL Image object).
    
    Returns:
        Parsed output from the model based on the task.
    
    Raises:
        ValueError: If required arguments (model, processor, image) are missing.
    """
    # Input validation
    if model is None or processor is None:
        raise ValueError("Model and processor must be provided.")
    if image is None:
        raise ValueError("An input image is required.")

    # Determine device (e.g., GPU or CPU) from the model
    device = model.device

    # Construct the full prompt
    prompt = task_prompt if text_input is None else task_prompt + text_input
    
    # Preprocess inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device, torch.float16)

    # Generate output
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    # Decode generated tokens
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process the output
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer