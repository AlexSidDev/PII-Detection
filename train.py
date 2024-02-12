from data_utils import create_dataset
from transformers import AutoTokenizer


if __name__ == '__main__':
    class_names = ['O', 'NAME_STUDENT', 'EMAIL', 'USERNAME', 'ID_NUM', 'PHONE_NUM', 'URL_PERSONAL', 'STREET_ADDRESS']

    labels_mapping = {class_name: ind for ind, class_name in enumerate(class_names)}

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    dataset = create_dataset('dataset/train.json', tokenizer, labels_mapping)
