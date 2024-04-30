# AutoTruss: Automatic Truss Design with Reinforcement Learning

The official repo of paper 'Automatic Truss Design with Reinforcement Learning'

## Script

Run the pipeline

```sh
python main.py --config 17_bar_case --run-id test
```

## Docker

Build container

```sh
docker build -t auto-truss .
```

Run container

```sh
docker run --rm --name auto-truss auto-truss:latest --config 17_bar_case --run-id test
```

## Add Customized Config

To generate your own truss with different configs, you can add your own config file in `configs` and register it in `config.py`.
