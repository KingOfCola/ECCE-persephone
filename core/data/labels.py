# -*-coding:utf-8 -*-
"""
@File    :   labels.py
@Time    :   2024/10/18 11:35:48
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Class for hashable labels
"""


class Label:
    def __init__(self, value: int | str, name: any):
        self.__value = value
        self.__name = name

    @property
    def value(self):
        return self.__value

    @property
    def name(self):
        return self.__name

    def __hash__(self):
        return hash(self.__value)

    def __eq__(self, other):
        if isinstance(other, Label):
            return self.__value == other.value
        return self.__value == other

    def __str__(self):
        return f"{self.__name} ({self.__value})"

    def __repr__(self):
        return str(self)
