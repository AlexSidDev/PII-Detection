from data_utils import create_dataset
from model_utils import create_model
from transformers import AutoTokenizer


if __name__ == '__main__':
    class_names = ['O', 'NAME_STUDENT', 'EMAIL', 'USERNAME', 'ID_NUM', 'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

    labels_mapping = {class_name: ind for ind, class_name in enumerate(class_names)}

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    dataset = create_dataset(['dataset/train.json', 'dataset/mixtral-8x7b-v1.json'],
                             tokenizer=tokenizer, labels_mapping=labels_mapping)

    model = create_model('dslim/distilbert-NER', labels_mapping)
