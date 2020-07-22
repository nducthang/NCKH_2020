import scrapy
from fptshop.items import FptshopItem


class QuotesSpider(scrapy.Spider):
    name = 'mobile'
    start_urls = ['https://fptshop.com.vn/dien-thoai?sort=ban-chay-nhat']
    home = 'https://fptshop.com.vn'

    def parse(self, response):
        finalPage = int(response.xpath('/html/body/section/div/div[2]/div[2]/div[4]/a[2]/@data-page').get())
        url_page = 'https://fptshop.com.vn/dien-thoai?sort=ban-chay-nhat&trang=1'
        print("Trang cuối:", finalPage)
        for page in range(finalPage):
            link = url_page.replace('1', str(page + 1))
            yield scrapy.Request(link, callback=self.craw_mobile)

    def craw_mobile(self, response):
        link_mobile = response.css('.fs-lpil .fs-lpil-img::attr(href)').extract()
        for link in link_mobile:
            yield scrapy.Request(self.home + link, callback=self.save_file)

    def save_file(self, response):
        item = FptshopItem()
        item['title'] = response.css('.fs-dttname ::text').get()
        item['screen'] = response.xpath('/html/body/section/div/div[3]/div[2]/div[1]/div[2]/ul/li[1]/span/text()').get()
        item['ram'] = response.xpath('/html/body/section/div/div[3]/div[2]/div[1]/div[2]/ul/li[4]/span/text()').get()
        item['link'] = response.url
        yield item
