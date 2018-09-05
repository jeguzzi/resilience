#!/usr/bin/env python

import click

from experiment import execute_all_experiments


@click.command()
@click.option('--config', default='experiment.yaml', help='The list of all experiments to compute')
@click.option('--pool', default=6, help='How many processors to use')
def main(config, pool):
    execute_all_experiments(config, pool=pool)


if __name__ == '__main__':
    main()
