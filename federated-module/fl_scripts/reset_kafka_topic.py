#!/usr/bin/env python3
"""
Script to delete and recreate a Kafka topic, ensuring it has 0 messages.
Usage: python reset_kafka_topic.py [topic_name]
If no topic_name is provided, the script will list all topics and ask for selection.
"""

import sys
import subprocess
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
NAMESPACE = 'kafkaml'

def get_kafka_pod():
    """Get the Kafka pod name"""
    try:
        result = subprocess.run([
            'kubectl', 'get', 'pods', '-n', NAMESPACE, '-o', 'jsonpath={.items[*].metadata.name}'
        ], capture_output=True, text=True, check=True)
        
        pods = result.stdout.strip().split()
        kafka_pod = None
        for pod in pods:
            if 'kafka' in pod.lower():
                kafka_pod = pod
                break
        
        if not kafka_pod:
            raise Exception("No Kafka pod found in namespace")
        
        logging.info(f"Found Kafka pod: {kafka_pod}")
        return kafka_pod
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get Kafka pod: {e}")
        raise

def list_all_topics(kafka_pod):
    """List all Kafka topics"""
    try:
        result = run_kafka_command(kafka_pod, [
            'kafka-topics.sh', '--bootstrap-server', KAFKA_BOOTSTRAP_SERVERS,
            '--list'
        ])
        
        topics = [topic.strip() for topic in result.split('\n') if topic.strip()]
        return topics
    except Exception as e:
        logging.error(f"Failed to list topics: {e}")
        return []

def get_topic_selection():
    """Get topic selection from user"""
    kafka_pod = get_kafka_pod()
    topics = list_all_topics(kafka_pod)
    
    if not topics:
        logging.error("No topics found in Kafka")
        return None
    
    print("\n" + "="*60)
    print("📋 AVAILABLE KAFKA TOPICS:")
    print("="*60)
    
    for i, topic in enumerate(topics, 1):
        print(f"{i:2d}. {topic}")
    
    print("="*60)
    
    while True:
        try:
            choice = input(f"\n🔢 Enter topic number (1-{len(topics)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                logging.info("User cancelled topic selection")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(topics):
                selected_topic = topics[choice_num - 1]
                print(f"\n✅ Selected topic: {selected_topic}")
                return selected_topic
            else:
                print(f"❌ Please enter a number between 1 and {len(topics)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return None

def run_kafka_command(kafka_pod, command):
    """Run a Kafka command inside the pod"""
    try:
        full_command = ['kubectl', 'exec', '-n', NAMESPACE, kafka_pod, '--'] + command
        result = subprocess.run(full_command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Kafka command failed: {e}")
        logging.error(f"Command: {' '.join(command)}")
        logging.error(f"Error output: {e.stderr}")
        raise

def topic_exists(kafka_pod, topic_name):
    """Check if topic exists"""
    try:
        result = run_kafka_command(kafka_pod, [
            'kafka-topics.sh', '--bootstrap-server', KAFKA_BOOTSTRAP_SERVERS,
            '--list'
        ])
        return topic_name in result
    except:
        return False

def delete_topic(kafka_pod, topic_name):
    """Delete the topic"""
    logging.info(f"Deleting topic: {topic_name}")
    try:
        run_kafka_command(kafka_pod, [
            'kafka-topics.sh', '--bootstrap-server', KAFKA_BOOTSTRAP_SERVERS,
            '--delete', '--topic', topic_name
        ])
        logging.info(f"✅ Topic {topic_name} deleted successfully")
        return True
    except Exception as e:
        logging.warning(f"Failed to delete topic {topic_name}: {e}")
        return False

def create_topic(kafka_pod, topic_name):
    """Create the topic"""
    logging.info(f"Creating topic: {topic_name}")
    try:
        run_kafka_command(kafka_pod, [
            'kafka-topics.sh', '--bootstrap-server', KAFKA_BOOTSTRAP_SERVERS,
            '--create', '--topic', topic_name,
            '--partitions', '1', '--replication-factor', '1'
        ])
        logging.info(f"✅ Topic {topic_name} created successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to create topic {topic_name}: {e}")
        return False

def get_message_count(kafka_pod, topic_name):
    """Get the message count in the topic"""
    try:
        result = run_kafka_command(kafka_pod, [
            'kafka-run-class.sh', 'kafka.tools.GetOffsetShell',
            '--broker-list', KAFKA_BOOTSTRAP_SERVERS,
            '--topic', topic_name, '--time', '-1'
        ])
        
        # Parse the output to get message count
        # Format: topic:partition:offset
        lines = result.strip().split('\n')
        total_messages = 0
        
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 3:
                    try:
                        offset = int(parts[2])
                        total_messages += offset
                    except ValueError:
                        continue
        
        return total_messages
    except Exception as e:
        logging.error(f"Failed to get message count: {e}")
        return -1

def verify_topic_empty(kafka_pod, topic_name):
    """Verify that the topic has 0 messages"""
    logging.info(f"Verifying topic {topic_name} has 0 messages...")
    
    # Wait a moment for topic to be fully created
    time.sleep(2)
    
    message_count = get_message_count(kafka_pod, topic_name)
    
    if message_count == 0:
        logging.info(f"✅ Topic {topic_name} verified: 0 messages")
        return True
    else:
        logging.error(f"❌ Topic {topic_name} has {message_count} messages (expected 0)")
        return False

def reset_kafka_topic(topic_name):
    """Main function to reset a Kafka topic"""
    try:
        logging.info(f"=== STARTING KAFKA TOPIC RESET FOR: {topic_name} ===")
        
        # Get Kafka pod
        kafka_pod = get_kafka_pod()
        
        # Check if topic exists
        if topic_exists(kafka_pod, topic_name):
            logging.info(f"Topic {topic_name} exists, proceeding with deletion...")
            delete_topic(kafka_pod, topic_name)
            # Wait for deletion to complete
            time.sleep(3)
        else:
            logging.info(f"Topic {topic_name} does not exist, skipping deletion...")
        
        # Create the topic
        if not create_topic(kafka_pod, topic_name):
            logging.error(f"Failed to create topic {topic_name}")
            return False
        
        # Verify topic is empty
        if not verify_topic_empty(kafka_pod, topic_name):
            logging.error(f"Topic {topic_name} verification failed")
            return False
        
        logging.info(f"=== KAFKA TOPIC RESET COMPLETED SUCCESSFULLY FOR: {topic_name} ===")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error resetting Kafka topic {topic_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def reset_single_topic(topic_name):
    """Reset a single topic with confirmation"""
    # Confirm the action with the user
    print(f"\n⚠️  WARNING: This will DELETE and RECREATE the topic '{topic_name}'")
    print("   All messages in this topic will be permanently lost!")
    
    while True:
        confirm = input(f"\n🤔 Are you sure you want to reset topic '{topic_name}'? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            break
        elif confirm in ['n', 'no', '']:
            logging.info("User cancelled the operation")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")
    
    success = reset_kafka_topic(topic_name)
    
    if success:
        logging.info("🎉 Topic reset completed successfully!")
        return True
    else:
        logging.error("💥 Topic reset failed!")
        return False

def interactive_mode():
    """Interactive mode that runs until user quits"""
    print("\n" + "="*60)
    print("🔄 KAFKA TOPIC RESET TOOL - INTERACTIVE MODE")
    print("="*60)
    print("This tool will help you reset Kafka topics.")
    print("You can reset multiple topics in one session.")
    print("Type 'q' or 'quit' at any time to exit.")
    print("="*60)
    
    while True:
        try:
            print("\n📋 What would you like to do?")
            print("1. List and select a topic to reset")
            print("2. Enter a topic name directly")
            print("3. Quit (q)")
            
            choice = input("\n🔢 Enter your choice (1-3) or 'q' to quit: ").strip().lower()
            
            if choice in ['q', 'quit', '3']:
                print("\n👋 Goodbye! Thanks for using the Kafka Topic Reset Tool!")
                break
            
            elif choice == '1':
                # Interactive topic selection
                topic_name = get_topic_selection()
                if topic_name:
                    reset_single_topic(topic_name)
                # Continue the loop for another operation
            
            elif choice == '2':
                # Direct topic name input
                topic_name = input("\n📝 Enter the topic name: ").strip()
                if topic_name:
                    if topic_exists(get_kafka_pod(), topic_name):
                        reset_single_topic(topic_name)
                    else:
                        print(f"❌ Topic '{topic_name}' does not exist!")
                        print("💡 Use option 1 to see available topics.")
                else:
                    print("❌ Topic name cannot be empty!")
                # Continue the loop for another operation
            
            else:
                print("❌ Please enter 1, 2, 3, or 'q'")
            
            # Ask if user wants to continue
            if choice in ['1', '2']:
                continue_choice = input("\n🔄 Would you like to reset another topic? (y/N): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("\n👋 Goodbye! Thanks for using the Kafka Topic Reset Tool!")
                    break
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Kafka Topic Reset Tool!")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"❌ An error occurred: {e}")
            continue_choice = input("\n🔄 Would you like to try again? (y/N): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break

def main():
    """Main entry point"""
    # Check if topic name was provided as command line argument
    if len(sys.argv) == 2:
        topic_name = sys.argv[1]
        if not topic_name:
            logging.error("Topic name cannot be empty")
            sys.exit(1)
        logging.info(f"Using topic from command line: {topic_name}")
        
        # Single topic reset mode
        success = reset_single_topic(topic_name)
        sys.exit(0 if success else 1)
    
    elif len(sys.argv) == 1:
        # Interactive mode
        interactive_mode()
        sys.exit(0)
    
    else:
        print("Usage: python reset_kafka_topic.py [topic_name]")
        print("Examples:")
        print("  python reset_kafka_topic.py FED-DEBUG-data_topic  # Single topic reset")
        print("  python reset_kafka_topic.py  # Interactive mode")
        sys.exit(1)

if __name__ == "__main__":
    main()
