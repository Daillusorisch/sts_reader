# sts_reader

Just because I think python is too slow

only provide one API for now: 

```python
def read_sts(path: str) -> tuple[np.ndarray, np.ndarray]: ...
```

and it assumes you're reading MAG data from MAVEN...