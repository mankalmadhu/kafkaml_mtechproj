#!/usr/bin/env python3
"""
Script to consume and display messages from a Kafka topic.
Usage: python check_kafka_topic.py [--bootstrap-servers SERVERS] [--topic TOPIC] [--from-beginning]
"""

import argparse
import json
import sys
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def consume_messages(bootstrap_servers, topic, from_beginning=False, max_messages=None):
    """
    Consume and display messages from a Kafka topic.
    
    Args:
        bootstrap_servers (str): Comma-separated list of Kafka bootstrap servers
        topic (str): Name of the topic to consume from
        from_beginning (bool): Whether to consume from the beginning or latest
        max_messages (int): Maximum number of messages to consume (None for unlimited)
    """
    try:
        # Configure auto_offset_reset based on from_beginning flag
        auto_offset_reset = 'earliest' if from_beginning else 'latest'
        
        logging.info(f"Connecting to Kafka at {bootstrap_servers}")
        logging.info(f"Topic: {topic}")
        logging.info(f"Reading from: {auto_offset_reset}")
        
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers.split(','),
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=False,
            group_id='manual-topic-checker',
            consumer_timeout_ms=10000  # 10 seconds timeout
        )
        
        logging.info("Successfully connected to Kafka. Waiting for messages...")
        logging.info("=" * 80)
        
        message_count = 0
        
        for msg in consumer:
            message_count += 1
            
            print(f"\n{'=' * 80}")
            print(f"Message #{message_count}")
            print(f"{'=' * 80}")
            print(f"Topic: {msg.topic}")
            print(f"Partition: {msg.partition}")
            print(f"Offset: {msg.offset}")
            print(f"Timestamp: {msg.timestamp}")
            print(f"Key (raw bytes): {msg.key}")
            
            # Try to decode key as integer (common pattern in this project)
            if msg.key:
                try:
                    key_as_int = int.from_bytes(msg.key, byteorder='big', signed=False)
                    print(f"Key (as int): {key_as_int}")
                except:
                    try:
                        print(f"Key (as string): {msg.key.decode('utf-8')}")
                    except:
                        print(f"Key (hex): {msg.key.hex()}")
            
            # Try to decode value as JSON
            print(f"\nValue:")
            if msg.value:
                try:
                    value_json = json.loads(msg.value)
                    print(json.dumps(value_json, indent=2))
                except:
                    try:
                        print(msg.value.decode('utf-8'))
                    except:
                        print(f"Raw bytes (first 500): {msg.value[:500]}")
                        print(f"Total size: {len(msg.value)} bytes")
            else:
                print("(empty)")
            
            print(f"{'=' * 80}\n")
            
            # Check if we've reached max messages
            if max_messages and message_count >= max_messages:
                logging.info(f"Reached maximum number of messages ({max_messages}). Stopping.")
                break
        
        logging.info(f"\nTotal messages consumed: {message_count}")
        
        if message_count == 0:
            logging.warning("No messages found in the topic (or timeout reached)")
        
        consumer.close()
        
    except KafkaError as e:
        logging.error(f"Kafka error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Consume and display messages from a Kafka topic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read latest messages from default topic
  python check_kafka_topic.py
  
  # Read from beginning
  python check_kafka_topic.py --from-beginning
  
  # Read first 10 messages
  python check_kafka_topic.py --from-beginning --max-messages 10
  
  # Specify custom bootstrap servers
  python check_kafka_topic.py --bootstrap-servers kafka:9092 --topic my-topic
        """
    )
    
    parser.add_argument(
        '--bootstrap-servers',
        default='kafka:9092',
        help='Comma-separated list of Kafka bootstrap servers (default: kafka:9092)'
    )
    
    parser.add_argument(
        '--topic',
        default='FED-pon3f18x-agg_control_topic',
        help='Kafka topic to consume from (default: FED-zga6v9aq-agg_control_topic)'
    )
    
    parser.add_argument(
        '--from-beginning',
        action='store_true',
        help='Start consuming from the beginning of the topic (default: latest)'
    )
    
    parser.add_argument(
        '--max-messages',
        type=int,
        default=None,
        help='Maximum number of messages to consume (default: unlimited)'
    )
    
    args = parser.parse_args()
    
    consume_messages(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        from_beginning=args.from_beginning,
        max_messages=args.max_messages
    )

if __name__ == '__main__':
    main()

