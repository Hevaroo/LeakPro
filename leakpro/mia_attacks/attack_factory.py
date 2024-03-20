from leakpro.mia_attacks.attacks.attack_p import AttackP
from leakpro.mia_attacks.attacks.rmia import AttackRMIA

from leakpro.mia_attacks.attack_objects import AttackObjects

class AttackFactory:
    attack_classes = {
        'attack_p': AttackP,
        'rmia': AttackRMIA,
    }

    @classmethod
    def create_attack(cls, name:str, attack_objects:AttackObjects, configs:dict):
        if name in cls.attack_classes:
            return cls.attack_classes[name](attack_objects, configs)
        else:
            raise ValueError(f"Unknown attack type: {name}")