from optparse import OptionParser


def get_args():
    parser = OptionParser()

    parser.add_option('--analysis-type', dest='analysis_type', default='dataset',
                      type='string', help='Type of Analysis: ["dataset", "custom_message"]')

    parser.add_option('--alpha', dest='alpha', default=1.,
                      type='float', help='Alpha')

    parser.add_option('--train-data-path', dest='train_data_path',
                      type='string', help='Train Data Path')
    parser.add_option('--test-data-path', dest='test_data_path',
                      type='string', help='Test Data Path')
    parser.add_option('--parameters-path', dest='parameters_path',
                      type='string', help='Parameters Path')

    parser.add_option('--custom-messages-path', dest='custom_messages_path',
                      type='string', help='Custom Message for Classifying')

    parser.add_option("--adversarial", dest="adversarial", action="store_true",
                      help="Analysing with Adversarial")

    (options, args) = parser.parse_args()
    return options
