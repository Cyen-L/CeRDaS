>>> uvicorn main:app --reload

>>> sudo docker build -t iris-img .
* For building new docker image
* '-t', flag tags the newly created image
* 'iris-ml-build', is the tag for the image
* '.', specifies the build context to be the current directory, includes all files and subdirectories

>>> sudo docker run -d -p 8000:80 --name iris-ctn iris-img

>>> sudo docker build -t iris-db:latest .

>>> sudo docker run -d --name iris-db-ctn -p 5432:5432 iris-db:latest

>>> sudo docker-compose up -d
* Start multiple containers as defined in a docker-compose.yml file in current directory
* Including PostgreSQL database

>>> sudo docker exec -it iris-db-ctn psql -U cy -d iris-db
* 'docker exec', tells Docker to execute a command in a running container
* '-it', flags allocate an interactive terminal.
psql -U myuser -d mydatabase: This invokes the PostgreSQL command-line interface (psql) as the user myuser to connect to the database mydatabase.

>>> sudo docker save -o iris-app-img.tar iris-app-img:latest
>>> sudo docker save -o postgres.tar postgres:latest

>>> sudo docker ps -a

>>> sudo docker images

>>> sudo docker stop iris-ctn

>>> sudo docker rm iris-ctn

>>> \l
* Lists all the databases in the PostgreSQL server

>>> \dt
* Lists all the tables in the currently connected database

>>> \d mytable

>>> SELECT Id, Model_Algo, Model_Parameters, Saved_Name, Model_Version, Balance_Accuracy, Generated_Time FROM Model;

>>> SELECT Id, Model_Id, Generated_Time, Output_Data FROM Prediction;