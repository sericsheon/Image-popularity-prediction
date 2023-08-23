from transformers import TFBertModel, BertTokenizer
import pandas as pd
import tensorflow as tf
import numpy as np 


df_pics = pd.read_csv("pics_final_model.csv")
df_earth = pd.read_csv("d_earth.csv")

caption_pics = df_pics['title'].astype(str).values
caption_earth = df_earth['title'].astype(str).values

input_text = caption_earth.tolist()

def get_embeddings(model_name, input_text):
    from transformers import TFAutoModel, AutoTokenizer
    import numpy as np

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = 200  # Process 200 sequences at a time
    batches = [input_text[i:i + batch_size] for i in range(0, len(input_text), batch_size)]
    embeddings = []

    # Process each batch
    for batch in batches:
        # Encode the batch
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='tf')

        # Extract embeddings
        with tf.device('/GPU:0'):  # specify the device you want to use
            model = TFAutoModel.from_pretrained(model_name)
            outputs = model(encoded_input)
            embeddings_batch = outputs[0][:,0,:].numpy()

        embeddings.append(embeddings_batch)

    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings


# Call the function for each model
embeddings_gpt2 = get_embeddings("gpt2", caption_earth.tolist())
# print(embeddings_gpt2.shape)
embeddings_bert = get_embeddings("bert-base-uncased", caption_earth.tolist())
# print(embeddings_bert.shape)
embeddings_xlnet = get_embeddings("xlnet-base-cased", caption_earth.tolist())
# print(embeddings_xlnet.shape)


np.save('gpt2_embeddings_earth.npy', embeddings_gpt2)
np.save('bert_embeddings_earth.npy', embeddings_bert)
np.save('xlnet_embeddings_earth.npy', embeddings_xlnet)