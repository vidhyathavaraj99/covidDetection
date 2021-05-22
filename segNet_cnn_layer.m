function seg_layertransform = segNet_cnn_layer(image)
%y-1 x-1
seg_layertransform=(image(2:end-1,2:end-1)>=image(1:end-2,1:end-2));
%y-1 x
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(1:end-2,2:end-1))*2;
%y-1 x+1
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(1:end-2,3:end))*4;
%y x-1
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(2:end-1,1:end-2))*8;
%y x+1
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(2:end-1,3:end))*16;
%y+1 x-1
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(3:end,1:end-2))*32;
%y+1 x
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(3:end,2:end-1))*64;
%y+1 x+1
seg_layertransform=seg_layertransform+(image(2:end-1,2:end-1)>=image(3:end,3:end))*128;
seg_layertransform=uint8(seg_layertransform);
end
