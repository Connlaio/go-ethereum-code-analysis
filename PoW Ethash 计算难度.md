# PoW Ethash 计算难度



```go
// CalcDifficulty is the difficulty adjustment algorithm. It returns
// the difficulty that a new block should have when created at time
// given the parent block's time and difficulty.
func (ethash *Ethash) CalcDifficulty(chain consensus.ChainHeaderReader, time uint64, parent *types.Header) *big.Int {
	return CalcDifficulty(chain.Config(), time, parent)
}

```

参数输入：

| 参数名 | 类型                        | 大小 | 说明                                                         |
| ------ | --------------------------- | ---- | ------------------------------------------------------------ |
| chain  | consensus.ChainHeaderReader |      | ChainHeaderReader defines a small collection of methods needed to access the local blockchain during header verification. |
| time   | uint64                      |      | 当前块的时间                                                 |
| parent | *types.Header               |      | 当前块的父节点                                               |

