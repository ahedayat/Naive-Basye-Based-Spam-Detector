import pandas as pd
import classifier as classifier
import dataloader as dataloader
import utils as utils
import adverserial as adversarial


def get_parameters(parameters_path):
    parameters = pd.read_pickle('{}'.format(parameters_path))
    parameters = parameters.set_index('par')['value'].to_dict()

    return parameters


def adversarial_message_generator(message, masks):
    new_message = list()
    for word in message:
        new_word = ''
        for ix, (letter) in enumerate(word):
            new_letter = letter
            if letter in masks:
                new_letter = masks[letter][0]
            new_word += new_letter
        new_message.append(new_word)
    return ' '.join(new_message)


def dataset_classify(detector, test_dataloader, adversarial_masks=None):

    # Classifying
    accuracy = 0
    adversarial_accuracy = 0

    for ix, (label, message) in enumerate(test_dataloader):
        _label = detector.classify(message)
        if _label == label:
            accuracy += 1

        if adversarial_masks is not None:
            adversarial_message = adversarial_message_generator(
                message, adversarial_masks)
            _adversarial_label = detector.classify(adversarial_message)
            if _adversarial_label == label:
                adversarial_accuracy += 1

        progress_percent = ((ix+1)/len(test_dataloader))*100
        print('Classifying: %d/%d ( %.2f %% )' %
              (ix, len(test_dataloader), progress_percent), end='\r')
    print()

    accuracy = accuracy / len(test_dataloader)
    adversarial_accuracy = adversarial_accuracy / len(test_dataloader)

    return len(test_dataloader), accuracy * 100, adversarial_accuracy * 100


def cleansing_message(message):
    message = message.replace('\W+', ' ').replace('\s+', ' ').strip()
    return message.lower().split()


def message_classify(detector, custom_messages_path, adversarial_masks=None):
    labels = list()
    adversarial_labels = list()
    for message in open(custom_messages_path):
        message = cleansing_message(message)
        label = detector.classify(message)

        adversarial_label = None
        if adversarial_masks is not None:
            adversarial_message = adversarial_message_generator(
                message, adversarial_masks)
            adversarial_label = detector.classify(adversarial_message)

            print('Message:')
            print(' '.join(message))

            print('Adversarial:')
            print(adversarial_message)
        adversarial_labels.append(adversarial_label)

        labels.append(label)

    return labels, adversarial_labels


def _main(args):

    assert args.analysis_type in [
        'dataset', 'custom_message'], 'Analysis Type must be: "dataset" or "custom_message".'

    # Parameters
    parameters = get_parameters(args.parameters_path)

    # Adversarial
    adversarial_masks = None
    if args.adversarial:
        adversarial_masks = adversarial.masks

    # Data Loaders
    print('Preparing Data Loaders...')
    train_dataloader = dataloader.SMS_SPAM(path=args.train_data_path)
    test_dataloader = dataloader.SMS_SPAM(path=args.test_data_path)

    # Detector
    print('Preparing Naive Bayes Classifier...')
    detector = classifier.Naive_Bayes(train_data=train_dataloader.data,
                                      parameters=parameters,
                                      alpha=args.alpha)

    if args.analysis_type == 'dataset':

        # Classifying Train Data
        print('{} Testing on Train Dataset {}'.format('#'*16, '#'*16))
        train_num_samples, train_accuracy, adversarial_train_accuracy = dataset_classify(detector=detector,
                                                                                         test_dataloader=train_dataloader,
                                                                                         adversarial_masks=adversarial_masks)

        # Classifying Test Data
        print('{} Testing on Test Dataset {}'.format('#'*16, '#'*16))
        test_num_samples, test_accuracy, adversarial_test_accuracy = dataset_classify(detector=detector,
                                                                                      test_dataloader=test_dataloader,
                                                                                      adversarial_masks=adversarial_masks)

        # Printing Results
        print('Accuracy on Train Data ({} Samples)-> (without adversarial):{}, (with adversarial):{}'.format(train_num_samples,
                                                                                                             train_accuracy,
                                                                                                             adversarial_train_accuracy))
        print('Accuracy on Test Data ({} Samples)-> (without adversarial):{}, (with adversarial):{}'.format(test_num_samples,
                                                                                                            test_accuracy,
                                                                                                            adversarial_test_accuracy))

    elif args.analysis_type == 'custom_message':
        labels, adversarial_labels = message_classify(detector=detector,
                                                      custom_messages_path=args.custom_messages_path,
                                                      adversarial_masks=adversarial_masks)

        print('Results: ')
        for ix, (label, adversarial_label) in enumerate(zip(labels, adversarial_labels)):
            print('{}. (without adversarial):{}, (with adversarial):{}'.format(
                ix+1, label, adversarial_label))

    else:
        print('There is an error...')
        exit(-1)


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
