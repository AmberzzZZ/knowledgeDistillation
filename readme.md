## biFPN

    upSamp: Upsample(resize) + 1x1 DCB(align)
    downSamp: maxpooling + 1x1 DCB
    same-level path: 3x3 DCB
    fuse: weight sum ('fast' mode)
    通道对齐: 1x1 conv-bn-relu

    depthConvBlock:
      ------ entry -----
      depth-wise conv: filters_in
      point-wise conv: filters_out
      BN
      ReLU
      ------ entry -----
      ------repeats by depth-------
      depth-wise conv: filters_out
      point-wise conv: filters_out
      BN
      ReLU
      ------repeats by depth-------  


    efficientDet的biFPN:
    upSamp: resize
    downSamp: maxpooling
    fuse: weight sum ('fast' mode)
    通道对齐: 1x1 conv-bn-swish