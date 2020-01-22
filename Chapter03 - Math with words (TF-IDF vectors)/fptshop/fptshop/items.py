# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FptshopItem(scrapy.Item):
    title = scrapy.Field()
    screen = scrapy.Field()
    ram = scrapy.Field()
    link = scrapy.Field()
