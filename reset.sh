sudo docker stop iris-app-ctn
sudo docker stop iris-db-ctn
sudo docker rm iris-app-ctn
sudo docker rm iris-db-ctn
sudo docker rmi iris-app-img
sudo docker rmi postgres
sudo docker-compose up