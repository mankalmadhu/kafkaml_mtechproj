from mainTraining import *
from callbacks import *

class SingleFederatedTraining(MainTraining):
    """Class for single models federated training
    
    Attributes:
        boostrap_servers (str): list of boostrap server for the Kafka connection
        result_url (str): URL for downloading the untrained model
        result_id (str): Result ID of the model
        control_topic(str): Control topic
        deployment_id (int): deployment ID of the application
        batch (int): Batch size used for training
        kwargs_fit (:obj:json): JSON with the arguments used for training
        kwargs_val (:obj:json): JSON with the arguments used for validation
    """

    def __init__(self):
        """Loads the environment information"""

        super().__init__()

        self.distributed = False
        self.incremental = False

        self.model_logger_topic, self.federated_string_id, self.agg_rounds, self.data_restriction, self.min_data, self.agg_strategy, self.registered_devices = load_federated_environment_vars()

        logging.info("Received federated environment information (model_logger_topic, federated_string_id, agg_rounds, data_restriction, min_data, agg_strategy) ([%s], [%s], [%d], [%s], [%d], [%s])",
                self.model_logger_topic, self.federated_string_id, self.agg_rounds, self.data_restriction, self.min_data, self.agg_strategy)
            
    def get_models(self):
        """Downloads the model and loads it"""

        super().get_single_model()

    def generate_and_send_data_standardization(self):
        """Generates and sends the data standardization"""
        
        super().generate_and_send_data_standardization(self.model_logger_topic, self.federated_string_id, self.data_restriction, self.min_data, self.distributed, self.incremental)

    def generate_federated_kafka_topics(self):
        """Generates the Kafka topics for the federated training"""

        super().generate_federated_kafka_topics()
    
    def parse_metrics(self, model_metrics):
        """Parse the metrics from the model"""

        return super().parse_metrics(model_metrics)

    def sendTempMetrics(self, train_metrics, val_metrics):
        """Send the metrics to the backend"""
        
        super().sendTempMetrics(train_metrics, val_metrics)

    def sendFinalMetrics(self, cf_generated, epoch_training_metrics, epoch_validation_metrics, test_metrics, dtime, cf_matrix):
        """Sends the metrics to the control topic"""

        self.stream_timeout = None
        
        return super().sendSingleMetrics(cf_generated, epoch_training_metrics, epoch_validation_metrics, test_metrics, dtime, cf_matrix)