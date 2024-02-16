>>> uvicorn main:app --reload

>>> sudo docker build -t iris-img .
* For building new docker image
* '-t', flag tags the newly created image
* 'iris-ml-build', is the tag for the image
* '.', specifies the build context to be the current directory, includes all files and subdirectories

>>> sudo docker run -d -p 8000:80 --name iris-ctn iris-img

>>> sudo docker ps -a

>>> sudo docker images

>>> sudo docker stop iris-ctn

>>> sudo docker rm iris-ctn