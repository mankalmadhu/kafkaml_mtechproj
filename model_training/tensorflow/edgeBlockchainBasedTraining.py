import time
import logging
import json
import traceback
import numpy as np

from decoders import *

from FederatedKafkaMLModelSink import FederatedKafkaMLModelSink
from KafkaModelEngine import KafkaModelEngine
from trigger_federated_training_job import trigger_federated_training_job

# TODO: Implement Incremental and Distributed Blockchain Federated Training if needed or going to be implemented
# from singleFederatedIncrementalTraining import SingleBlockchainFederatedIncrementalTraining
# from distributedFederatedTraining import DistributedBlockchainFederatedTraining
# from distributedFederatedIncrementalTraining import DistributedBlockchainFederatedIncrementalTraining

def aggregate_model(model, trained_model, aggregation_strategy, control_msg):
  """Aggregates the model with the trained model"""

  if aggregation_strategy == 'FedAvg':
    weights = [model.get_weights(), trained_model.get_weights()]
    new_weights = list()
    for weights_list_tuple in zip(*weights): 
        new_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
      
    model.set_weights(new_weights)

  elif aggregation_strategy == 'FedAvg+':
    # Weighted FedAvg: se hace un promedio de los modelos, pero se le da mas peso a los modelos mas recientes
    raise NotImplementedError
  elif aggregation_strategy == 'Another':
    raise NotImplementedError
  else:
    raise NotImplementedError('Aggregation strategy not implemented')
  
  return model


def convert_numpy_types(obj):
  """Convert NumPy types to Python native types for JSON serialization"""
  if isinstance(obj, np.floating):
      return float(obj)
  elif isinstance(obj, np.integer):
      return int(obj)
  elif isinstance(obj, np.ndarray):
      return obj.tolist()
  elif isinstance(obj, dict):
      return {key: convert_numpy_types(value) for key, value in obj.items()}
  elif isinstance(obj, list):
      return [convert_numpy_types(item) for item in obj]
  else:
      return obj


def aggregate_client_trained_models(global_model, client_trained_models, aggregation_strategy, control_msg):
  """
  Aggregate client trained models using the specified aggregation strategy.
  Simplified - just passes model weights, no metadata needed.
  """
  from aggregation_strategies import get_aggregation_strategy
  
  if not client_trained_models:
    logging.warning("No client models to aggregate")
    return global_model
  
  logging.info("Aggregating {} client models using {} strategy".format(len(client_trained_models), aggregation_strategy))
  
  # Get the aggregation strategy instance
  strategy = get_aggregation_strategy(aggregation_strategy)
  
  # Extract weights from each trained model
  device_weights = [model.get_weights() for model in client_trained_models]
  
  # Aggregate using the simplified API
  aggregated_weights = strategy.aggregate(global_model.get_weights(), device_weights)
  
  # Set the aggregated weights back to the global model
  global_model.set_weights(aggregated_weights)
  
  logging.info("Batch aggregation completed using {}".format(aggregation_strategy))
  
  return global_model


def _extract_metric(metrics, section, keys):
  section_data = metrics.get(section, {}) if isinstance(metrics, dict) else {}
  for key in keys:
    if not isinstance(section_data, dict):
      continue
    values = section_data.get(key)
    if isinstance(values, list) and values:
      return values[-1]
    if values is not None and not isinstance(values, list):
      return values
  return None


def _round_float(value, ndigits=4):
  if value is None:
    return None
  if isinstance(value, bool):
    return value
  try:
    float_value = float(value)
  except (TypeError, ValueError):
    return value
  return round(float_value, ndigits)


def _safe_int(value):
  try:
    return int(value)
  except (TypeError, ValueError):
    return value


def EdgeBlockchainBasedTraining(training):
    training.get_models()
    """Downloads the models from the URLs received, saves and loads them from the filesystem to Tensorflow models"""

    # TODO: Implement Incremental and Distributed Blockchain Federated Training if needed
    # if isinstance(training, DistributedFederatedTraining) or isinstance(training, DistributedFederatedIncrementalTraining):
    #   training.configure_distributed_models()
    # """Distributed models configuration"""
    
    training.generate_and_send_data_standardization()
    """Generates the data standardization and sends it to the model control topic"""

    training.generate_federated_kafka_topics()
    """Generates the federated Kafka topics to receive the data from the federated nodes"""
  
    training_settings = {'batch': training.batch, 'kwargs_fit': training.kwargs_fit, 'kwargs_val': training.kwargs_val}
    
    # TODO: Implement Incremental and Distributed Blockchain Federated Training if needed
    # if isinstance(training, SingleFederatedIncrementalTraining):
    #   training_settings['stream_timeout'] = training.stream_timeout
    #   training_settings['monitoring_metric'] = training.monitoring_metric
    #   training_settings['change'] = training.change
    #   training_settings['improvement'] = training.improvement
    # elif isinstance(training, DistributedFederatedTraining):
    #   training_settings['optimizer'] = training.optimizer
    #   training_settings['learning_rate'] = training.learning_rate
    #   training_settings['loss'] = training.loss
    #   training_settings['metrics'] = training.metrics
    # elif isinstance(training, DistributedFederatedIncrementalTraining):
    #   training_settings['stream_timeout'] = training.stream_timeout
    #   training_settings['monitoring_metric'] = training.monitoring_metric
    #   training_settings['change'] = training.change
    #   training_settings['improvement'] = training.improvement
    #   training_settings['optimizer'] = training.optimizer
    #   training_settings['learning_rate'] = training.learning_rate
    #   training_settings['loss'] = training.loss
    #   training_settings['metrics'] = training.metrics

    rounds, model_metrics, start_time = 0, [], time.time()
    last_client_model_topic = None
    clients_contributions = dict()
    client_trained_models = []
    aggration_round_metrics = []
    """Initializes the version, rounds, model metrics and start time"""

    sink = FederatedKafkaMLModelSink(bootstrap_servers=training.bootstrap_servers, topic=training.model_data_topic, control_topic=training.model_control_topic, federated_id=training.result_id, training_settings=training_settings)
    
    isNextAggRound = True

    while rounds < training.agg_rounds:
      logging.info("Round: {}".format(rounds))

      if isNextAggRound:
        control_msg = sink.send_model(training.model, rounds)
        logging.info("Model sent to Federated devices")
        
        if rounds == 0:
          logging.info("Control message: %s", control_msg)
          training_settings = json.dumps(convert_numpy_types(control_msg['training_settings']))
          model_compile_args = json.dumps(convert_numpy_types(control_msg['model_compile_args']))
          training.save_model_architecture(training_settings, control_msg['model_architecture'], model_compile_args)
        else:
          training.save_update_along_with_global_model(last_client_model_topic, control_msg['topic'])

        training.write_control_message_into_blockchain(control_msg['topic'], rounds)
        logging.info("Waiting for Federated devices to send their models and blockchain confirmation")      
      
      while training.elements_to_aggregate() < 1:
        continue
      
      logging.info("A federated device sent its model. Aggregating models")

      last_client_model_topic, metrics, client_account, client_data_size = training.retrieve_last_model_from_queue()


      # add last_client_model_topic to the logging.info below
      logging.info("Model received from client [%s], data size [%s] at round [%s] with metrics [%s] and client model topic [%s]", client_account, client_data_size, rounds, metrics, last_client_model_topic)

      try:
        control_msg = {
                        'topic': last_client_model_topic,
                        'metrics': metrics,
                        'account': client_account,
                        'num_data': client_data_size
                      }
        
        logging.info("Message received for prediction")

        model_reader = KafkaModelEngine(training.bootstrap_servers, 'server')
        trained_model = model_reader.setWeights(training.model, control_msg)
        logging.info("Model received from Federated devices")
        
        if training.agg_strategy == 'FedAvg':
          training.model = aggregate_model(training.model, trained_model, training.agg_strategy, control_msg)
        
        client_trained_models.append(trained_model)
        logging.info("Added device model to client_trained_models. Now contains %d models for round %d", len(client_trained_models), rounds)
        
        model_metrics.append(metrics)
 
        train_metrics, val_metrics = training.parse_metrics(model_metrics)
        training.sendTempMetrics(train_metrics, val_metrics)
        """ Sends the current metrics to the backend"""

        clients_contributions[client_account] = client_data_size
        reward = training.calculate_reward(rounds, control_msg, clients_contributions)

        aggration_round_metric = {
          'round': rounds,
          'topic': last_client_model_topic,
          'account': client_account,
          'num_samples': _safe_int(client_data_size),
          'train_loss': _round_float(_extract_metric(metrics, 'training', ['loss'])),
          'train_acc': _round_float(_extract_metric(metrics, 'training', ['accuracy', 'binary_accuracy'])),
          'val_loss': _round_float(_extract_metric(metrics, 'validation', ['loss'])),
          'val_acc': _round_float(_extract_metric(metrics, 'validation', ['accuracy', 'binary_accuracy'])),
          'reward': _round_float(reward)
        }
        aggration_round_metrics.append(aggration_round_metric)

        
        # Enhanced wait condition: check registered_devices count if enabled
        if training.registered_devices != -1 and len(client_trained_models) < training.registered_devices:
            while training.elements_to_aggregate() < 1:
              logging.info("Waiting for more client models to aggregate for round %s", rounds)
              time.sleep(1)
            
            logging.info("Queue has %d items. Continuing to process. client_trained_models: %d/%d models", 
                        training.elements_to_aggregate(), len(client_trained_models), training.registered_devices)
            isNextAggRound = False
            continue

 
        logging.info("Proceeding to aggregate. client_trained_models containing %d models (registered_devices: %d, queue items: %d)", 
                    len(client_trained_models), training.registered_devices, training.elements_to_aggregate())
        if training.agg_strategy != 'FedAvg':
          training.model = aggregate_client_trained_models(training.model, client_trained_models, training.agg_strategy, control_msg)
        
        rounds += 1
        isNextAggRound = True
        client_trained_models = []
        logging.info("Aggregation completed. New model version: {}".format(rounds))

      except Exception as e:
        traceback.print_exc()
        logging.error("Error with the received data [%s]. Waiting for new a new prediction.", str(e))

    # Saving last round update and sending final model
    if rounds > 0:
      training.save_update_along_with_global_model(last_client_model_topic, control_msg['topic'])
      training.write_control_message_into_blockchain(control_msg['topic'], rounds)

    logging.info("Federated training finished. Sending final model to Kafka-ML Cloud and stopping smart contract")
    sink.close()

    training.send_stopTraining()

    training.reward_participants()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Total training time: %s", str(elapsed_time))

    logging.info("Taking last metrics per epoch")
    
    train_metrics, val_metrics = training.parse_metrics(model_metrics)
    logging.info("Epoch training metrics: %s", str(train_metrics))
    logging.info("Epoch validation metrics: %s", str(val_metrics))

    training.sendFinalMetrics(False, train_metrics, val_metrics, [], elapsed_time, None, aggration_round_metrics)
    logging.info("Sending final model and metrics to Kafka-ML Cloud")
    """Sends the final metrics to the backend"""

    logging.info("Edge-based training (%s) finished", type(training).__name__)