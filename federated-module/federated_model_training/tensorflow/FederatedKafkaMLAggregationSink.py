__author__ = 'Cristian Martin Fdez'

from kafka import KafkaConsumer, TopicPartition, KafkaProducer
import struct
import logging
import json
import pickle

class FederatedKafkaMLAggregationSink(object):
    """Class representing a sink of federated learning data to Apache Kafka. This class will allow to receive 
        federated learning data in the send methods and will send it through Apache Kafka. 
        Once all data would have been sent to Kafka, and the `close` method a 
        message will be sent to the federated topic. 

    Args:
        bootstrap_servers (str): List of Kafka brokers
        topic (str): Kafka topic 
        federated_id (str): Federated ID for sending the data
        control_topic (str): Control Kafka topic for sending confirmation after sending training data. 
            Defaults to federated
        group_id (str): Group ID of the Kafka consumer. Defaults to federated

    Attributes:
        bootstrap_servers (str): List of Kafka brokers
        topic (str): Kafka topic 
        federated_id (str): Federated ID for sending the data
        control_topic (str): Control Kafka topic for sending confirmation after sending training data. 
            Defaults to federated
        group_id (str): Group ID of the Kafka consumer. Defaults to federated
        __partitions (:obj:dic): list of partitions and offsets of the self.topic received
        __consumer (:obj:KafkaConsumer): Kafka consumer
        __producer (:obj:KafkaProducer): Kafka producer
    """

    def __init__(self, bootstrap_servers, topic, control_topic, 
                 federated_id, data_standard=None, training_settings=None, group_id='federated'):

        logging.info("=== STARTING FederatedKafkaMLAggregationSink CONSTRUCTOR ===")
        logging.info("Constructor parameters:")
        logging.info("  - bootstrap_servers: %s", bootstrap_servers)
        logging.info("  - topic: %s", topic)
        logging.info("  - control_topic: %s", control_topic)
        logging.info("  - federated_id: %s", federated_id)
        logging.info("  - data_standard: %s", data_standard)
        logging.info("  - training_settings: %s", training_settings)
        logging.info("  - group_id: %s", group_id)

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.control_topic = control_topic
        self.federated_id = federated_id
        self.data_standard = data_standard if data_standard is not None else {'input_shape': None, 'output_shape': None, 'labels': None}
        self.training_settings = training_settings if training_settings is not None else {'batch': None, 'kwargs_fit': None, 'kwargs_val': None, 'test_rate': None}

        self.group_id = group_id


        self.total_messages = 0

        self.__partitions = {}

        logging.info("Creating KafkaConsumer for partitions...")
        self.__consumer = KafkaConsumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False            
        )
        logging.info("✅ KafkaConsumer created successfully")

        logging.info("Creating KafkaProducer...")
        self.__producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            max_request_size= 2**31-1
        )
        logging.info("✅ KafkaProducer created successfully")

        logging.info("About to call __init_partitions()...")
        self.__init_partitions()
        logging.info("✅ __init_partitions() completed successfully")
        logging.info("Partitions received [%s]", str(self.__partitions))
        logging.info("=== FederatedKafkaMLAggregationSink CONSTRUCTOR COMPLETED ===")

    def __object_to_bytes(self, value):
        """Converts a python type to bytes. Types allowed are string, int, bool and float."""
        if value is None:
            return None
        elif value.__class__.__name__ == 'bytes':
            return value
        elif value.__class__.__name__ in ['int', 'bool']:
            return bytes([value])
        elif value.__class__.__name__ == 'list':
            return pickle.dumps(value)
        elif value.__class__.__name__ == 'float':
            return struct.pack("f", value)
        elif value.__class__.__name__ == 'str':
            return value.encode('utf-8')
        elif value.__class__.__name__ == 'dict':
            return json.dumps(value).encode('utf-8')
        elif value.__class__.__name__ == 'ndarray':
            return pickle.dumps(value)
        else:
            logging.error('Type %s not supported for converting to bytes', value.__class__.__name__)
            raise Exception("Type not supported")
            
    def __get_partitions_and_offsets(self):
        """Obtains the partitions and offsets in the topic defined"""
        
        logging.info("=== STARTING __get_partitions_and_offsets ===")
        logging.info("Topic: %s", self.topic)
        logging.info("Consumer: %s", self.__consumer)
        
        dic = {} 
        logging.info("Calling partitions_for_topic(%s)...", self.topic)
        partitions = self.__consumer.partitions_for_topic(self.topic)
        logging.info("Partitions received: %s", partitions)
        
        if partitions is not None:
            logging.info("Processing %d partitions...", len(partitions))
            for p in self.__consumer.partitions_for_topic(self.topic):
                logging.info("Processing partition %d...", p)
                tp = TopicPartition(self.topic, p)
                logging.info("Created TopicPartition: %s", tp)
                
                logging.info("Assigning partition %s to consumer...", tp)
                self.__consumer.assign([tp])
                logging.info("✅ Partition assigned successfully")
                
                logging.info("Seeking to end of partition %s...", tp)
                self.__consumer.seek_to_end(tp)
                logging.info("✅ Seek to end completed")
                
                logging.info("Getting position for partition %s...", tp)
                last_offset = self.__consumer.position(tp,100) or 0
                logging.info("✅ Position obtained: %d", last_offset)
                
                dic[tp.partition] = {'offset': last_offset}
                logging.info("✅ Partition %d processed, offset: %d", tp.partition, last_offset)
        else:
            logging.warning("No partitions found for topic %s", self.topic)
        
        logging.info("Final partitions dictionary: %s", dic)
        logging.info("=== __get_partitions_and_offsets COMPLETED ===")
        return dic
    
    def __init_partitions(self):
        """Obtains the partitions and offsets in the topic defined and save them in self.__partitions"""
        
        logging.info("=== STARTING __init_partitions ===")
        self.__partitions = self.__get_partitions_and_offsets()
        logging.info("✅ __init_partitions completed successfully")
        logging.info("=== __init_partitions COMPLETED ===")

    def __update_partitions(self):
        """Updates the offsets and length in the topic defined after sending data"""
        
        dic = self.__get_partitions_and_offsets()
        self.total_messages = 0
        for p in dic.keys():
            if p in self.__partitions.keys():
                diff = dic[p]['offset'] - self.__partitions[p]['offset']
                self.__partitions[p]['length'] = dic[p]['offset']
            else:
                diff = dic[p]['offset']
                self.__partitions[p] = {'offset': 0, 'length': dic[p]['offset']}
            self.total_messages += diff
        
        logging.info("%d messages have been sent to Kafka", self.total_messages)
    
    def __stringify_partitions(self):
        """Stringify the partition information for sending to Apache Kafka"""
        
        res = ""
        for p in self.__partitions.keys():
            """ Format topic:partition:offset:length,topic1:partition:offset:length"""
            res+= self.topic +":"+ str(p)+ ":" + str(self.__partitions[p]['offset'])
            if 'length' in self.__partitions[p]:
                res += ":" + str(self.__partitions[p]['length'])
            res+=","
        
        res = res[:-1]
        """Remove last ',' list of topics"""

        return res

    def __send_control_msg(self, model):
        """Sends control message to Apache Kafka with the information"""

        dic = {
            'topic': self.__stringify_partitions(),
            'data_standard': self.data_standard,
            'training_settings': self.training_settings,
            'model_architecture': model.to_json()
        }
        key = self.__object_to_bytes(self.federated_id)
        data = json.dumps(dic).encode('utf-8')
        self.__producer.send(self.control_topic, key=key, value=data)
        self.__producer.flush()
        logging.info("Control message to Kafka %s", str(dic))

    def __send_control_msg_metrics(self, metrics, version, num_data, record_metadata):
        """Sends control message to Apache Kafka with the information"""

        offset = record_metadata[0].offset
        length = len(record_metadata) + offset
        topic_str = f'{self.topic}:{record_metadata[0].partition}:{offset}:{length}'
        dic = {
            'topic': topic_str,
            'version': version+1 if version != -1 else -1,
            'metrics': metrics,
            'num_data': num_data
        }
        key = self.__object_to_bytes(self.federated_id)
        data = json.dumps(dic).encode('utf-8')
        self.__producer.send(self.control_topic, key=key, value=data)
        self.__producer.flush()
        logging.info("Control message to Kafka %s", str(dic))

        return dic

    def __send(self, data, label=None):
        """Converts data and label received to bytes and sends them to Apache Kafka"""
        
        data = self.__object_to_bytes(data)
        label = self.__object_to_bytes(label)
        if label is None:
            return self.__producer.send(self.topic, value=data)
        else:
            return self.__producer.send(self.topic, key=label, value=data)

    def send_model(self, model):
        """Sends layer weights"""
        self.__init_partitions()
        for idx, layer_w in enumerate(model.get_weights()):
            # if len(model.layers[i]) > 0:
            logging.info("Sending layer %d, shape %s", idx, str(layer_w.shape))
            self.__send(data=layer_w, label=idx)    
            self.__producer.flush()

        self.__update_partitions()
        self.__send_control_msg(model)

    def send_model_and_metrics(self, model, metrics, version, num_data):
        self.__init_partitions()
        record_metadata = []
        for idx, layer_w in enumerate(model.get_weights()):
            # if len(model.layers[i]) > 0:
            logging.info("Sending layer %d, shape %s", idx, str(layer_w.shape))
            feature = self.__send(data=layer_w, label=idx)
            record_metadata.append(feature.get(timeout=10))       
            self.__producer.flush()

        self.__update_partitions()
        control_msg = self.__send_control_msg_metrics(metrics, version, num_data, record_metadata)

        return control_msg
        
    def close(self):
        """Closes the connection with Kafka and sends the control message to the control topic"""

        # self.__producer.flush()
        # self.__update_partitions()
        # self.__send_control_msg()
        self.__producer.close()
        self.__consumer.close(autocommit=False)