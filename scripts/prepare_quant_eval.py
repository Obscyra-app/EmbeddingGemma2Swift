"""Fixed synthetic dev and holdout; no training/calibration or personal content."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
dev=[
('The train departs from the north station at 08:40. Tickets include one suitcase.','Когда отправляется поезд с северного вокзала?','What luggage is included in the train fare?'),
('Return unopened headphones within fourteen days. A receipt is required for a refund.','Можно ли вернуть нераспакованные наушники?','What proof is needed for a headphone refund?'),
('The bicycle workshop replaces brake pads and straightens bent wheels on Saturdays.','Где починить тормоза велосипеда в субботу?','Where can I fix a bent bicycle wheel?'),
('Back up the database before migration. Restore the snapshot if the schema update fails.','Как восстановить базу при неудачной миграции?','What should happen before updating the database schema?'),
('Indoor basil needs sunlight and lightly moist soil. Avoid leaving water in the tray.','Как поливать базилик на подоконнике?','How much light does indoor basil need?'),
('This apartment lease allows one cat. Smoking inside and subletting are prohibited.','Разрешает ли договор аренды держать кошку?','Can I sublet this apartment?'),
('To reset the router, hold its rear button for ten seconds. Saved network settings will be erased.','Как сбросить настройки роутера?','What does a router factory reset erase?'),
('The museum is closed on Mondays. Children under six enter free with an adult.','В какой день музей закрыт?','Do five-year-old children need a museum ticket?'),
('For pancakes, mix flour, milk and eggs, then cook thin portions on a hot pan.','Как приготовить тонкие блины?','Which ingredients are used for pancakes?'),
('The parcel weighs two kilograms. It will arrive at the collection point on Friday.','Когда посылка приедет в пункт выдачи?','How heavy is the parcel?'),
('Reduce video file size by lowering the bitrate. Keep a copy of the original footage.','Как уменьшить размер видеофайла?','Should I preserve the original video before compression?'),
('The camping checklist includes a tent, sleeping bag, headlamp and drinking water.','Что взять с собой для ночёвки в палатке?','Which light belongs on the camping checklist?')]
holdout=[
('Conference badges can be collected at the west entrance from nine in the morning.','Где получить бейдж участника конференции?','When does conference badge collection open?'),
('The printer accepts A4 paper only. Clear the rear paper path before restarting after a jam.','Как убрать замятую бумагу из принтера?','Which paper size does this printer accept?'),
('The swimming pool requires a cap. Outdoor shoes must be left in the changing room.','Нужна ли шапочка для бассейна?','Where should outdoor shoes be left at the pool?'),
('The astronomy club meets at the observatory during clear nights to view Saturn through a telescope.','Где клуб астрономов наблюдает Сатурн?','What weather is needed for the telescope meeting?'),
('Charging the wireless mouse takes ninety minutes. A blinking amber light indicates a low battery.','Что означает мигающий жёлтый индикатор мыши?','How long does charging the wireless mouse take?'),
('Store the watercolor paintings flat and away from direct sunlight to prevent the colors fading.','Как хранить акварельные рисунки?','Why should watercolor paintings stay out of sunlight?'),
('The rehearsal studio has a drum kit and two guitar amplifiers. Book at least one day in advance.','Есть ли в репетиционной студии барабаны?','How early should the rehearsal studio be reserved?'),
('The ferry carries pedestrians and bicycles but no cars. Crossings take twenty minutes.','Можно ли перевезти автомобиль на этом пароме?','How long is the ferry crossing?')]
for split,rows in [('dev',dev),('holdout',holdout)]:
 docs=[r[0] for r in rows];queries=[{'text':q,'relevant':i} for i,r in enumerate(rows) for q in r[1:]]
 data={'split':split,'scope':'author-written synthetic bilingual retrieval, not a general benchmark','documents':docs,'queries':queries}
 (a.output/f'{split}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 request={'texts':['title: none | text: '+d for d in docs]+['task: search result | query: '+q['text'] for q in queries]}
 (a.output/f'{split}-request.json').write_text(json.dumps(request,ensure_ascii=False)+'\n')
# Fresh held-out image fixtures; selection must not inspect their outputs.
for name,color,shape in [('yellow-circle',(245,215,20),'circle'),('purple-square',(140,40,190),'square'),('orange-triangle',(240,120,15),'triangle')]:
 im=Image.new('RGB',(216,168),'white');d=ImageDraw.Draw(im)
 if shape=='circle':d.ellipse((48,24,168,144),fill=color)
 elif shape=='square':d.rectangle((48,24,168,144),fill=color)
 else:d.polygon([(108,20),(180,145),(36,145)],fill=color)
 im.save(a.output/f'{name}.png')
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.iterdir())}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
