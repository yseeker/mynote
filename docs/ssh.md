# リモートでJupyterlabを開く

```bash
ssh -L <host port>:localhost:<remote port> user@remote

docker run -it --rm -p <host port>:<remote port> --name <container-name> -v:$PWD:/work <image-name> /bin/bash

jupyter lab --ip 0.0.0.0 --port <container port> --allow-root
```