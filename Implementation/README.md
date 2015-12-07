1. run get_models.sh to get the CNN caffemodel 

2. setup torch7
  
  run the following in command:
  
  cd ~/
  curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
  git clone https://github.com/torch/distro.git ~/torch --recursive
  cd ~/torch; ./install.sh

3. Install loadcaffe
  run the following in command:

  sudo apt-get install libprotobuf-dev protobuf-compiler
  luarocks install loadcaffe
  
4.  run the following in command:
  th neural_style.lua -content_image content_image.jpg -style_image style_image.jpg
