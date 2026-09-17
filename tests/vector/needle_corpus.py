"""A small real corpus, so some of the vectors under test are real embeddings.

Synthetic vectors are clustered on purpose, but their geometry is still an
assumption. These passages are embedded through a real model and buried in a
knowledge base that is otherwise synthetic noise, which is the case that matters:
an approximate index either surfaces a genuine answer from among unrelated
neighbours or it quietly does not.

`PAIRS` are passages with a question that a reader would expect to retrieve
them. The question never repeats the passage's wording wholesale, so retrieval
has to come from meaning rather than from overlap. `DISTRACTORS` fill the rest
of the collection with plausible prose on unrelated subjects.
"""

# (passage, question)
PAIRS = [
    (
        'The freezing point of pure water at one atmosphere is zero degrees Celsius, but dissolved salt lowers it: '
        'seawater at average ocean salinity stays liquid down to about minus 1.9 degrees.',
        'Why does the sea not freeze at the same temperature as a lake?',
    ),
    (
        'A capacitor stores energy in the electric field between two conductors. Its capacitance grows with the '
        'plate area and falls with the distance separating the plates.',
        'What makes one capacitor hold more charge than another at the same voltage?',
    ),
    (
        'Sourdough rises because wild yeasts in the starter produce carbon dioxide while lactic acid bacteria '
        'acidify the dough, which strengthens the gluten network and slows staling.',
        'What gives naturally leavened bread its lift and its keeping quality?',
    ),
    (
        'The Antonine Wall ran for about sixty kilometres between the Firth of Forth and the Firth of Clyde. It was '
        'built of turf on a stone foundation and was abandoned roughly twenty years after completion.',
        'Which Roman frontier in Scotland was made of turf rather than stone, and how long did it last?',
    ),
    (
        'In a B-tree every leaf sits at the same depth, so a lookup costs the same number of page reads wherever the '
        'key falls. Splits propagate upward only when a node is full.',
        'Why is the cost of finding a row by primary key so predictable?',
    ),
    (
        'Cyanobacteria released oxygen into an atmosphere that had almost none, and the dissolved iron in the oceans '
        'precipitated as banded iron formations before free oxygen could accumulate in the air.',
        'What happened to the oxygen produced by the earliest photosynthesis?',
    ),
    (
        'A fugue states a subject in one voice, answers it in another at the dominant, and then develops both '
        'through episodes that modulate before the final entries return to the tonic.',
        'How is a piece of counterpoint in this strict form usually laid out?',
    ),
    (
        'Aluminium is protected by a thin oxide layer that reforms instantly when scratched. Mercury destroys that '
        'layer, which is why mercury is barred from aircraft cabins.',
        'Why is a spilled thermometer considered dangerous on an aeroplane?',
    ),
    (
        'The Coriolis effect deflects moving air to the right in the northern hemisphere and to the left in the '
        'southern, which is why cyclones rotate in opposite directions either side of the equator.',
        'What decides which way a storm system spins?',
    ),
    (
        'Parmesan is aged for at least twelve months, during which the protein casein breaks down into free amino '
        'acids, including the glutamate that gives the cheese its savoury depth.',
        'Where does the strong savoury taste of a long-matured hard cheese come from?',
    ),
    (
        'A two-stroke engine completes its cycle in one crankshaft revolution by using the crankcase to pre-compress '
        'the incoming charge, so it fires twice as often as a four-stroke of the same speed.',
        'How can a small engine produce a power stroke on every turn?',
    ),
    (
        'Lichens are a partnership between a fungus that provides structure and moisture and an alga or '
        'cyanobacterium that photosynthesises. Neither partner survives alone in the same habitats.',
        'What kind of organism is actually two species living together on bare rock?',
    ),
    (
        'The Julian calendar assumed a year of exactly 365.25 days, overshooting the tropical year by eleven '
        'minutes. By the sixteenth century the equinox had drifted ten days from its nominal date.',
        'Why was the calendar reformed in 1582?',
    ),
    (
        'Copy-on-write lets two processes share the same physical pages until one of them writes, at which point the '
        'kernel duplicates only the page being modified.',
        'How does forking a large process avoid duplicating all of its memory?',
    ),
    (
        'Tea leaves for oolong are bruised at the edges so that oxidation begins there while the centre stays green, '
        'giving a leaf that is partly oxidised rather than uniformly so.',
        'What processing step produces a tea that sits between green and black?',
    ),
    (
        'The Mariana Trench reaches nearly eleven kilometres below sea level, where the pressure is over a thousand '
        'times that at the surface and the water temperature stays a few degrees above freezing.',
        'How deep is the lowest point of the ocean, and what are conditions like there?',
    ),
    (
        'Hashing with linear probing keeps colliding entries in nearby slots, which is cache-friendly, but clusters '
        'grow and lookups degrade sharply once the load factor passes about seventy per cent.',
        'What goes wrong with an open-addressed table as it fills up?',
    ),
    (
        'The Basque language has no demonstrated relationship to any other living language, and its grammar marks '
        'the agent of a transitive verb with a case absent from its neighbours.',
        'Which European language has no known relatives?',
    ),
    (
        'Nitrogen-fixing bacteria in root nodules convert atmospheric nitrogen into ammonia using an enzyme that is '
        'destroyed by oxygen, so the nodule maintains a low-oxygen interior.',
        'How do legumes obtain nitrogen without fertiliser?',
    ),
    (
        'A whip cracks because the tapering of the lash concentrates the same momentum into less and less mass, '
        'until the tip exceeds the speed of sound and produces a small sonic boom.',
        'What is the sharp noise made by a whip?',
    ),
    (
        'Byzantine gold coinage held its weight and fineness for seven centuries, which made it the reference '
        'currency of Mediterranean trade long after the western empire had gone.',
        'Which medieval coin was trusted across the Mediterranean for centuries?',
    ),
    (
        'Write-ahead logging requires that the log record describing a change reaches stable storage before the '
        'modified data page does, so a crash can always be replayed forward.',
        'What rule lets a database recover cleanly after losing power?',
    ),
    (
        'Chameleons change colour by rearranging a lattice of guanine nanocrystals in their skin, which shifts the '
        'wavelength the lattice reflects rather than moving pigment.',
        'How does a lizard shift its colour without changing pigment?',
    ),
    (
        'A geostationary satellite orbits at about 35 786 kilometres, where the orbital period matches one sidereal '
        'day, so it appears fixed above one point on the equator.',
        'Why must a television satellite sit at one particular altitude?',
    ),
    (
        'In Japanese joinery the kanawa splice locks two beams end to end with interlocking tongues and a driven '
        'key, carrying tension without glue or metal fasteners.',
        'How are two timbers joined lengthwise without nails or adhesive?',
    ),
    (
        'Grass burns differently from wood because its silica content leaves a glassy ash, which is why straw-fired '
        'boilers foul their heat exchangers faster than log-fired ones.',
        'Why do straw-burning boilers need cleaning more often?',
    ),
    (
        'The placebo response is measurable in conditions with a subjective endpoint, such as pain or nausea, and '
        'largely absent where the endpoint is a laboratory value like a tumour size.',
        'When does an inert treatment actually show an effect in a trial?',
    ),
    (
        'Shipping containers are stacked nine high on deck and lashed with rods; the corner castings, not the walls, '
        'carry the entire load of the boxes above.',
        'What part of a freight box bears the weight of the ones stacked on it?',
    ),
    (
        'The Doppler shift of a receding galaxy grows with its distance, and the constant relating the two gives an '
        'estimate of the age of the universe when extrapolated backwards.',
        'How is the age of the cosmos estimated from starlight?',
    ),
    (
        'Sea ice rejects most of its salt as it freezes, leaving brine channels that drain downward, so multi-year '
        'ice is fresh enough to drink once melted.',
        'Why can old pack ice be melted for drinking water?',
    ),
    (
        'Garbage collectors that move objects must update every reference to them, which is why languages with '
        'unrestricted pointer arithmetic cannot use a compacting collector.',
        'Why can some languages never compact their heap?',
    ),
    (
        'The Silk Road carried paper westward long before it carried silk eastward in quantity; papermaking reached '
        'Samarkand in the eighth century and Europe some four hundred years later.',
        'How did the technique of making paper travel out of China?',
    ),
    (
        'Reinforced concrete works because steel and concrete expand at nearly the same rate with temperature, so '
        'the bond between them survives the seasons.',
        'What coincidence of materials makes concrete reinforcement practical?',
    ),
    (
        'Bats emitting constant-frequency calls compensate for their own flight speed by lowering the emitted pitch, '
        'keeping the returning echo inside the narrow band their ears are tuned to.',
        'How does a flying animal keep its echoes in a usable frequency range?',
    ),
    (
        'A perfect fifth in equal temperament is two cents narrower than the pure ratio of three to two, a '
        'compromise that lets a keyboard play in every key.',
        'What is sacrificed to let an instrument play equally well in all keys?',
    ),
    (
        'Peat bogs preserve bodies because the sphagnum releases an acid that tans skin while binding the calcium '
        'that bone needs, so the skin survives and the skeleton dissolves.',
        'Why do ancient remains found in wetlands keep their skin but lose their bones?',
    ),
    (
        'Read committed isolation takes a fresh snapshot for each statement, so two identical queries in one '
        'transaction can legitimately return different rows.',
        'Why might the same query give different answers inside one transaction?',
    ),
    (
        'Saffron is the dried stigma of a sterile autumn crocus that must be propagated by dividing corms, and each '
        'flower yields three threads picked by hand.',
        'What makes this particular spice so expensive to produce?',
    ),
    (
        'A tuned mass damper near the top of a tall building swings out of phase with the structure, converting sway '
        'into heat in its own dampers rather than motion the occupants feel.',
        'How is the swaying of a skyscraper reduced for the people inside?',
    ),
    (
        'Icelandic turf houses used the earth itself as insulation, with the timber frame kept dry by a roof pitched '
        'steeply enough to shed rain before it soaked through.',
        'How were dwellings kept warm where there was almost no wood or coal?',
    ),
    (
        'The eye of a hurricane is warm and nearly cloudless because air subsides there, and the strongest winds sit '
        'in the ring of convection immediately around it.',
        'Why is it briefly calm at the centre of a tropical storm?',
    ),
    (
        'Gradual typing lets annotated and unannotated code coexist by inserting checks at the boundary, so the '
        'guarantees hold only where the annotations reach.',
        'What are the limits of adding type annotations to an existing dynamic codebase?',
    ),
    (
        'Whale falls support a succession of communities for decades: scavengers first, then enrichment feeders in '
        'the sediment, and finally bacteria living off the lipids in the bones.',
        'What happens on the sea floor after a large animal dies and sinks?',
    ),
    (
        'Damascus steel patterns come from bands of carbide that survive forging, and reproducing them depends on '
        'trace vanadium in the original ore rather than on the smith alone.',
        'Why has the surface pattern of these historical blades been hard to reproduce?',
    ),
    (
        'The Marshall Islands stick charts encoded swell refraction around atolls rather than the positions of the '
        'islands themselves, and were memorised before a voyage rather than carried on it.',
        'What did Pacific navigators actually record on their navigational devices?',
    ),
    (
        'Enzymes lower activation energy by stabilising the transition state, not the substrate; binding the '
        'substrate too tightly would make a catalyst worse rather than better.',
        'Why is it a mistake for a catalyst to hold its target too firmly?',
    ),
    (
        'Sorting networks fix their comparisons in advance regardless of the data, which makes them slower on '
        'average but suitable for hardware where branching is expensive.',
        'When is a data-independent sorting method preferable?',
    ),
    (
        'Venetian glassmakers were confined to Murano partly to contain furnace fires and partly to keep their '
        'techniques from leaving the republic, and emigration was punished severely.',
        'Why was an entire craft moved to one island?',
    ),
]

DISTRACTORS = [
    'The postal service introduced a sorting code in the 1960s, dividing the country into districts that machines '
    'could read directly from the envelope.',
    'Hedgerows planted as field boundaries now serve as wildlife corridors, linking woodland fragments that would '
    'otherwise be isolated from one another.',
    'Bicycle gearing is expressed as development, the distance travelled per pedal revolution, which lets wheels of '
    'different sizes be compared fairly.',
    'A jet of water entrains surrounding air, which is why a narrow nozzle can move far more air than passes '
    'through it.',
    'The first commercial refrigerated ships carried beef from Argentina, changing what European cities ate within '
    'a single generation.',
    'Clay tablets were kept damp while a scribe worked and then either left to dry or fired deliberately when the '
    'record mattered.',
    'Mountain roads use hairpins because a vehicle can climb a steeper grade in short sections than it can sustain '
    'continuously without overheating.',
    'The oboe gives the tuning note in an orchestra because its pitch is the least affected by temperature changes '
    'in the hall.',
    'Glacial erratics are boulders carried far from their parent rock, and mapping them was how the extent of past '
    'ice sheets was first established.',
    'Cast iron pans hold heat well but conduct it poorly, so they take longer to warm and keep hot spots where the '
    'burner sits.',
    'Semaphore towers relayed messages across France faster than a horse, but only in daylight and only in clear '
    'weather.',
    'The rings of a felled tree record drought years as narrow bands, and matching those bands across timbers dates '
    'buildings precisely.',
    'Sand used in construction cannot come from deserts: wind-rounded grains do not interlock the way river sand does.',
    'Lighthouse lenses were built in rings of prisms so that a thin sheet of glass could bend light that a solid '
    'lens of the same power could not carry.',
    'The gauge of a railway determines the tightest curve it can take, which is why mountain lines were often built '
    'narrower than main lines.',
    'Cheese rinds may be washed, bloomed or simply dried, and each treatment selects a different set of surface '
    'organisms.',
    'Airships used helium in the United States and hydrogen elsewhere, largely because the only economic source of '
    'helium was American natural gas.',
    'Bookbinding sewn on tapes opens flatter than one glued at the spine, which matters for anything meant to lie '
    'open on a desk.',
    'Ancient roads followed ridgelines where the ground was drier, and the valley routes came later with drainage '
    'and bridges.',
    'The tempering of chocolate aligns cocoa butter into one crystal form, and the snap of a well-tempered bar '
    'comes from that alignment.',
    'Wool insulates when wet because its fibres trap air even after absorbing a third of their weight in water.',
    'Harbour walls are built with gaps so that waves lose energy in turbulence rather than reflecting back into the '
    'approach.',
    'Type foundries cast metal letters in an alloy of lead, tin and antimony; the antimony expands slightly on '
    'cooling and fills the mould.',
    'Bees maintain the brood nest near thirty-five degrees by clustering or fanning, regardless of the weather '
    'outside.',
    'Canals need a summit level fed by reservoirs, because every boat passing a lock sends a lockful of water '
    'downhill.',
    'Photographic film responds to blue light more strongly than to red, which is why early portraits show skies '
    'as blank white.',
    'The mechanical escapement releases a clock train one step at a time, and its design sets how much the '
    'timekeeping depends on the driving force.',
    'Terracing turns a slope into a series of level fields, slowing runoff enough that soil stays where it was put.',
    'Stained glass is coloured in the melt by metal oxides, so the colour goes right through the sheet rather than '
    'sitting on its surface.',
    'The keel of a sailing boat resists sideways motion, which is what lets the hull convert wind across the beam '
    'into forward travel.',
    'Charcoal burns hotter than the wood it came from because the volatile fraction has already been driven off.',
    'Roman concrete used volcanic ash that continued reacting with seawater, so marine structures gained strength '
    'where modern ones lose it.',
    'Bell founders tune a bell by turning metal from the inside, which lowers particular partials without changing '
    'the overall shape.',
    'Lacquer hardens by absorbing moisture from the air rather than drying, so it is cured in a humid cabinet.',
    'The strings of a piano are struck at a point roughly one seventh along their length, which suppresses a '
    'partial that would otherwise sound harsh.',
    'Peat cut for fuel was stacked to dry in the wind, since a wet turf carries more water than it does energy.',
    'Fishing nets are dyed to match the water because a net visible against the light is avoided by the shoal.',
    'Windmill sails were shaped with a twist along their length so that each part met the wind at a usable angle.',
    'Vellum is prepared by stretching skin wet and scraping it as it dries, which is what gives the sheet its '
    'tension and smoothness.',
    'Freight moves by water where speed does not matter, because the power needed to push a hull rises with the '
    'cube of its speed.',
    'A dry stone wall is built in two leaves tied together by through stones, and it stands because the weight '
    'settles inward rather than outward.',
    'Coppiced woodland yields poles on a cycle of several years, and the stools themselves can outlive any tree '
    'left to grow normally.',
    'Ink for fountain pens must stay liquid in the feed but dry quickly on paper, a balance struck with humectants '
    'and surfactants.',
    'The angle of a plough turns the furrow slice over completely, burying weeds where they cannot reach light.',
    'Rope laid in three strands holds together because each strand is twisted against the lay of the whole.',
    'Malting germinates barley just far enough to develop the enzymes, then halts it with heat before the grain '
    'spends its starch.',
    'Sundials must be cut for the latitude where they stand, since the gnomon has to point at the celestial pole.',
    'Watermills on a fast stream used an undershot wheel, while those on a slow one needed a pond and an overshot '
    'wheel to gain head.',
    'Thatch sheds water through the pitch of the roof rather than by any seal, and the reed nearest the ridge wears '
    'out first.',
    'Tanning with oak bark takes months, and the resulting leather is stiff, thick and suited to soles rather than '
    'to clothing.',
    'Salt pans work by letting sun and wind remove the water, and the first crystals to form are discarded because '
    'they carry the wrong minerals.',
    'Longbows were made from a single stave taking in both heartwood and sapwood, so one face resisted compression '
    'while the other resisted tension.',
    'Lime mortar sets slowly by reabsorbing carbon dioxide, which lets a wall move slightly without cracking.',
    'Hop varieties differ in their bittering acids and their aromatic oils, and the two are added at different '
    'points in the boil.',
    'Ship rigging was tarred against rot, and the standing rigging was tarred more heavily than the running rigging '
    'that had to pass through blocks.',
    'The pitch of an aircraft propeller is set so that the blade meets the air at a useful angle along its whole '
    'length despite the tip moving faster.',
    'Ice houses were dug into north-facing slopes and packed with straw, keeping a summer supply from winter ponds.',
    'Wattle and daub uses woven sticks as a key for a mixture of clay, sand and straw, and the straw controls '
    'shrinkage as it dries.',
    'Grafting joins a chosen variety to a rootstock selected for vigour and soil tolerance, so one tree carries two '
    'sets of characteristics.',
    'Gold leaf is beaten between sheets of skin until it is a fraction of a micron thick, at which point it '
    'transmits green light.',
    'Church bells were rung in changes rather than tunes because a swinging bell cannot be started and stopped quickly enough to follow a melody.',
    'Cork is stripped from the living tree every nine years or so, and the first harvest is too irregular to use for bottle closures.',
    'Marble weathers by dissolution rather than by flaking, so carved detail on an exposed statue softens gradually instead of breaking away.',
    'Wrought iron contains threads of slag that give it a grain, and a smith works along that grain much as a joiner works along wood.',
    'Kilns for firing pottery were often built into a slope so that the draught carried heat up through the stacked wares.',
    'Sailmakers cut cloth in panels aligned with the expected load, since the fabric stretches far more on the bias than along the weave.',
    'Hedge laying part-cuts each stem and bends it over, so the plant survives and thickens into a barrier that a fence cannot match.',
    'Verdigris pigment was made by exposing copper to vinegar fumes, and it degrades in oil paint unless sealed from the binder.',
    'Millstones were dressed with a pattern of furrows that both cut the grain and swept the flour outward as the stone turned.',
    'Alpine pastures are grazed in a strict order through the season, moving upward as the snow retreats and down again before it returns.',
    'Wooden ship hulls were sheathed in copper to stop shipworm, which meant the iron fastenings had to be replaced to avoid galvanic corrosion.',
    'Papyrus sheets were made by laying strips in two directions and pressing them, the plant sap serving as its own adhesive.',
    'Brick clamps fired unevenly, producing hard bricks near the fire and soft ones at the edges, and builders sorted them by where they would be used.',
    'The bloomery process never melted iron; it produced a spongy mass that had to be hammered to expel the slag trapped inside.',
    'Coastal fog supports forests where rainfall alone could not, because the trees comb moisture directly out of the moving air.',
    'Snowshoes work by spreading load, and their length is chosen for the snow conditions rather than for the weight of the wearer alone.',
    'Bowyers season a stave for years before working it, since wood that moves after the bow is tillered will spoil the draw.',
    'Fire lookouts were sited so that any two of them could triangulate a smoke column, which mattered more than the view from any one of them.',
    'Mediterranean terraces retain heat as well as soil, and vines on the lowest terrace ripen days ahead of those above.',
    'Baskets woven from split willow last longer than those from whole rods, because the split face dries and sets against its neighbour.',
    'The strength of a knot comes from friction at the crossings, and a knot that is strong in one rope may slip in another of different construction.',
    'Bogs accumulate peat only where waterlogging stops decay, so drainage for agriculture releases carbon stored over thousands of years.',
    'Clocks on ships had to survive motion and temperature change, which is why a pendulum was useless at sea and a balance spring was not.',
    'Wine grapes grown on poor soil produce smaller berries with a higher skin to juice ratio, concentrating colour and tannin.',
    'Roof slates are hung with a head lap so that water crossing a joint in one course meets solid slate in the course below.',
    'Silk is reeled from the cocoon in a single continuous filament, and the cocoon must be treated before the moth breaks that filament on emerging.',
    'Ploughing across a slope rather than up and down it keeps furrows from becoming channels that carry soil to the bottom of the field.',
    'Iron gall ink bites into the paper surface, which makes it difficult to erase but also means it can eventually eat through the sheet.',
    'Stone circles were often built with one stone deliberately taller, and the alignment to that stone rather than the circle carries the astronomy.',
    'Chimneys draw because hot gas is less dense than the air outside, so the taller the flue the stronger the pull.',
    'Drystone field walls in upland country were built from stone cleared off the fields themselves, which is why their height follows the local geology.',
    'Cider apples are chosen for tannin and acid rather than for eating, and a single variety rarely makes a balanced drink on its own.',
    'Woodwind bores are either cylindrical or conical, and that choice decides which harmonics the tube will produce when overblown.',
    'Flood meadows were deliberately irrigated in late winter, the moving water warming the soil and depositing silt before the spring growth.',
    'Anchor cable was measured in shackles so that the amount paid out could be called across a deck without ambiguity.',
    'Blacksmiths judge temperature by colour, and the working range for forging is narrower than the range in which the metal merely glows.',
    'Reed beds filter waste water by supplying oxygen to root-zone bacteria, and the reeds themselves remove comparatively little.',
    'Cheese caves keep humidity high so that rinds develop slowly; a dry cave produces a cracked rind and a wasted wheel.',
    'Timber framing marks each joint with matching numerals cut before the frame is dismantled, so that it can be raised in the right order.',
    'Sled dogs are harnessed in pairs or in a fan depending on the terrain, since trees make a wide formation impossible.',
    'Wool was graded by the length of the staple long before it was graded by fineness, because length decided whether it could be combed or only carded.',
    'Church roofs were leaded with sheets cast on a bed of sand, and the sand texture on the underside identifies the casting method centuries later.',
    'Beer was historically brewed twice from the same grain, the second running producing a weak drink safe to give to children.',
    'The draught of a canal boat was limited by the depth of the cut, and an overloaded boat would drag and slow the whole passage.',
    'Hand planes are set with the mouth close to the blade for difficult grain, since a tight mouth breaks the shaving before it can tear.',
    'Alum was needed to fix dyes, and control of its supply shaped trade routes well beyond the textile trade itself.',
    'Orchards were planted with a pollinator variety scattered among the main crop, since many fruit trees set poorly with their own pollen.',
    'Fishermen read the colour of water to find a shoal, since plankton that feeds the fish also changes how the surface reflects light.',
    'Roads built on peat were floated on brushwood mattresses, spreading the load rather than trying to reach firm ground below.',
    'Bell ropes have a woven sally so that the ringer can find the grip by touch without looking up.',
    'Salt was taxed by weight, which encouraged drying it as completely as possible before it left the works.',
    'Glass blown by hand carries slight variation in thickness, and old window panes show it as distortion rather than as any flow of the glass.',
    'Sheep were folded on arable land overnight specifically for their manure, and the flock was valued for that as much as for wool.',
    'Cathedral vaults transfer thrust outward as well as downward, which is what the buttresses along the flank exist to resist.',
    'Kiln-dried timber moves less in service than air-dried, but the drying schedule has to be slow enough to avoid case hardening.',
    'Ferries crossing a tidal race run on the slack water, and a timetable that ignores the tide is a timetable that cannot be kept.',
    'Charcoal burners lived beside their stacks for days because a stack that burns through rather than smouldering is a lost week of work.',
    'Weaving sheds were built with north-facing sawtooth roofs so that the light on the looms stayed even through the day.',
    'Hives were sited facing the morning sun so that the colony started foraging early, which matters most in a short northern season.',
    'Riverbanks were revetted with willow because the cuttings root, so the structure becomes stronger rather than weaker with time.',
    'Signal flags carry meaning by combination as well as by individual letter, and a single flag can stand for a complete instruction.',
    'Lead crystal takes deep cutting because the metal oxide softens the glass, and the same oxide raises its refractive index.',
    'Waterproof cloth was once made by soaking it in linseed oil, which cures into a flexible film but remains prone to self-heating in a heap.',
    'Flint was worked by removing flakes from a prepared core, and the shape of the core determined the shape of every flake taken from it.',
    'Cattle droving routes kept to wide verges so that the herd could graze while it travelled, and those verges survive as green lanes.',
    'Frost hollows form where cold air drains into a valley bottom and cannot escape, so the lowest ground is the last to be planted.',
    'Carpets knotted by hand have a pile that leans one way, and the same carpet looks lighter or darker depending on which end you view it from.',
    'Barrels are made without any fastening between staves, held entirely by the hoops and the swelling of the wood once wet.',
    'Sugar was refined by repeated crystallisation, and the darker syrups left behind were sold separately rather than discarded.',
    'A mill leat had to be cut with a fall small enough to avoid scouring but large enough to keep the channel from silting.',
    'Mountain huts store fuel rather than food, because fuel cannot be carried up in an emergency but food often can.',
    'Fences on moorland are built to be overtopped by drifting snow without collapsing, since a rigid fence simply catches the drift.',
    'Lime was burnt in kilns close to the quarry, since quicklime is lighter than the stone and more valuable to transport.',
    'Bread ovens were fired and then swept out, the loaves baking on the stored heat of the masonry rather than over a live fire.',
    'Rivers were gauged at a narrow section with a stable bed, because a reading taken where the channel shifts means nothing over time.',
    'Hazel was coppiced on a seven-year cycle for hurdles, and a stool left uncut too long produces poles too thick to weave.',
    'Church clocks often had one hand, since the nearest quarter hour was as much precision as the day required.',
    'Peat ash was spread as a potash fertiliser, and the practice depleted the very bogs that supplied the fuel.',
    'Boats built clinker fashion overlap their planks, which makes a hull that flexes; carvel building butts them for a smooth rigid shell.',
    'Vines are pruned to a set number of buds, because a vine left to itself sets more fruit than it can ripen.',
    'Tallow candles guttered and smelled, while beeswax burned cleanly, and the difference decided which was used in a church and which at home.',
    'Harvest was timed by the grain rather than the calendar, and cutting slightly early avoided losing the crop to shedding.',
    'Drainage tiles were laid without joints sealed, since water was meant to enter through the gaps along the whole run.',
    'Sea walls fail at their ends rather than their faces, because the wave energy concentrates where the defence stops.',
    'Wool grease recovered from scouring was worth enough that mills processed the washings rather than discharging them.',
    'Turnpikes charged by the width of the wheel, on the theory that a broad wheel rolled the road while a narrow one cut it.',
    'Stone was quarried along its natural bed and had to be laid the same way, or it would delaminate in the face of the building.',
    'Nets were barked in a tannin solution to slow rotting, and the practice tinted them the colour that gave the fishery its look.',
    'Cellars were vaulted in brick because brick tolerates damp better than timber and needs no ventilation to survive it.',
    'Flax was retted by soaking until the stem tissue broke down, and over-retting weakened the fibre the process was meant to release.',
    'Horses were shod differently for road and field, the road shoe harder and the field shoe shaped to release mud.',
]
