"""
Kafka topic administration for KafkaML E2E automation
"""

import logging
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

logger = logging.getLogger(__name__)


class KafkaAdmin:
    """Kafka topic management"""
    
    def __init__(self, bootstrap_servers: str):
        """
        Initialize Kafka admin client
        
        Args:
            bootstrap_servers: Kafka bootstrap servers (e.g., 'localhost:9094')
        """
        self.bootstrap_servers = bootstrap_servers
    
    def ensure_topic_exists(self, topic_name: str, num_partitions: int = 1, replication_factor: int = 1) -> bool:
        """
        Ensure a Kafka topic exists, create it if it doesn't
        
        Args:
            topic_name: Name of the topic
            num_partitions: Number of partitions (default: 1)
            replication_factor: Replication factor (default: 1)
            
        Returns:
            True if topic exists or was created successfully
        """
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='kafkaml-e2e-admin'
            )
            
            # Check if topic exists
            existing_topics = admin_client.list_topics()
            
            if topic_name in existing_topics:
                logger.info(f"Topic '{topic_name}' already exists")
                admin_client.close()
                return True
            
            # Create topic
            logger.info(f"Creating topic '{topic_name}' with {num_partitions} partition(s)...")
            topic = NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )
            
            admin_client.create_topics(new_topics=[topic], validate_only=False)
            logger.info(f"✓ Topic '{topic_name}' created successfully")
            
            admin_client.close()
            return True
            
        except TopicAlreadyExistsError:
            logger.info(f"Topic '{topic_name}' already exists (race condition)")
            return True
        except Exception as e:
            logger.error(f"Failed to create topic '{topic_name}': {e}")
            return False

