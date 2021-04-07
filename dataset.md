# DataSet





DAG和epoch

1、上面的dataset就来自内存中的一组数据或者硬盘里的DAG。
2、DAG是有向无环图，以太坊的DAG是基于区块高度生成的。
3、以太坊中每3万个块会生成一代DAG，这一代就成称为一个epoch。
4、挖矿的时候需要从DAG中随机选取dataset，所以挖矿工作只能在现世DAG创建以后才能开始。