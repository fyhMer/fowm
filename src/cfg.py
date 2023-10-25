import os
import re
from omegaconf import OmegaConf


def parse_cfg(cfg_path: str, **kwargs) -> OmegaConf:
	"""Parses a config file and returns an OmegaConf object."""
	base = OmegaConf.load(cfg_path / 'config.yaml')
	cli = OmegaConf.from_cli()
	for k,v in cli.items():
		if v == None:
			cli[k] = True
	base.merge_with(cli)

	# Modality config
	if cli.get('modality', base.modality) not in {'state', 'pixels', 'all'}:
		raise ValueError('Invalid modality: {}'.format(cli.get('modality', base.modality)))
	modality = cli.get('modality', base.modality)
	if modality != 'state':
		mode = OmegaConf.load(cfg_path / f'pixels.yaml')
		base.merge_with(mode, cli)

	# Task config
	if base.task.startswith('xarm'):
		domain = base.task
	else:
		try:
			domain, task = base.task.split('-', 1)
		except:
			raise ValueError(f'Invalid task name: {base.task}')
	domain_path = cfg_path / 'tasks' / f'{domain}.yaml'
	assert os.path.exists(domain_path), f"No config file for `{domain}`"
	print(f"Using task config file '{domain_path}'")
	domain_cfg = OmegaConf.load(domain_path)
	base.merge_with(domain_cfg, cli)

	# Algebraic expressions
	for k in base.keys():
		try:
			v = base[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					base[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(base[k], float) and base[k].is_integer():
						base[k] = int(base[k])
		except:
			pass

	# Convenience
	base.task_title = base.task.replace('-', ' ').title()
	base.exp_name = str(base.get('exp_name', 'default'))

	return base
