## VM SSH Setup


- Create an ssh key in your local system in the `.ssh` folder - [Guide](https://cloud.google.com/compute/docs/connect/create-ssh-keys#linux-and-macos)

- Add the public key (`.pub`) to your VM instance - [Guide](https://cloud.google.com/compute/docs/connect/add-ssh-keys#expandable-2)

- Create a config file in your `.ssh` folder

  ```bash
  touch ~/.ssh/config
  ```

- Copy the following snippet and replace with External IP of the Kafka, Spark (Master Node), Airflow VMs. Username and path to the ssh private key

    ```bash
    Host Musicdata-Streaming-Pipeline-kafka
        HostName <External IP Address>
        User <username>
        IdentityFile <path/to/home/.ssh/keyfile>

    Host Musicdata-Streaming-Pipeline-spark
        HostName <External IP Address Of Master Node>
        User <username>
        IdentityFile <path/to/home/.ssh/keyfile>

    Host Musicdata-Streaming-Pipeline-airflow
        HostName <External IP Address>
        User <username>
        IdentityFile <path/to/home/.ssh/gcp>
    ```

- Once you are setup, you can simply SSH into the servers using the below commands in separate terminals. Do not forget to change the IP address of VM restarts.

    ```bash
    ssh Musicdata-Streaming-Pipeline-kafka
    ```

    ```bash
    ssh Musicdata-Streaming-Pipeline-spark
    ```

    ```bash
    ssh Musicdata-Streaming-Pipeline-airflow
    ```

- You will have to forward ports from your VM to your local machine for you to be able to see Kafka, Airflow UI. Check how to do that.

