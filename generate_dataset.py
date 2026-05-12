import csv
import os
import random
from collections import Counter

random.seed(42)

# Each entry: (response_text, sentiment, adoption_stage, main_barrier)
# Template-driven: labels are derived from the combination, not assigned independently.
# This guarantees every row has ground-truth labels that match its text.
TEMPLATES = [

    # ── POSITIVE / SCALING / NONE ─────────────────────────────────────────────
    ("We've fully embedded AI into our operations and the impact on productivity has been enormous. I'd recommend it to any SME willing to put in the work.", "positive", "scaling", "none"),
    ("AI is now central to how we run the business — customer queries, stock forecasting, invoicing. The ROI has been outstanding.", "positive", "scaling", "none"),
    ("After a year of scaling our AI tools, revenue is up and headcount has stayed flat. Best investment we've made.", "positive", "scaling", "none"),
    ("We went all-in on AI twelve months ago and haven't looked back. Our team is more focused and our customers are happier.", "positive", "scaling", "none"),
    ("AI handles our scheduling, email triage, and basic customer support. We've grown 40% without adding staff.", "positive", "scaling", "none"),
    ("Expanding AI across every department has transformed our workflows. We're more agile than companies twice our size.", "positive", "scaling", "none"),
    ("We're now running AI for demand forecasting, segmentation, and content creation. Each use case pays for itself.", "positive", "scaling", "none"),
    ("Our AI rollout has gone better than expected. The whole team is bought in and we keep finding new applications.", "positive", "scaling", "none"),

    # ── POSITIVE / SCALING / COST ─────────────────────────────────────────────
    ("Yes, scaling AI has cost us money — but the productivity gains dwarf the investment. We're more profitable because of it.", "positive", "scaling", "cost"),
    ("The licensing fees are real, but so are the savings. Our AI tools have paid for themselves three times over this year.", "positive", "scaling", "cost"),
    ("Cost was our biggest concern when we started expanding AI. A year on, the numbers speak for themselves.", "positive", "scaling", "cost"),
    ("We stretched the budget to scale AI properly and it's paid off. The ongoing costs are just part of our operating model now.", "positive", "scaling", "cost"),
    ("Not cheap, but not a luxury either. Scaling AI has made us lean and competitive in ways that justify every pound spent.", "positive", "scaling", "cost"),
    ("The monthly spend is significant but the efficiency gains more than offset it. We'd scale further if we could.", "positive", "scaling", "cost"),
    ("We had a thorough cost-benefit analysis before scaling and the numbers stacked up. Still happy with that decision.", "positive", "scaling", "cost"),
    ("It's expensive technology, but we treat it like any other capital investment — and the returns are there.", "positive", "scaling", "cost"),

    # ── POSITIVE / SCALING / SKILLS ───────────────────────────────────────────
    ("We invested in training alongside the AI rollout and it's paid off. The team is confident and the tools keep improving.", "positive", "scaling", "skills"),
    ("Skills were a gap when we started scaling AI, but we upskilled the team and now they're driving adoption themselves.", "positive", "scaling", "skills"),
    ("We brought in a part-time AI specialist to help with the rollout. Cost more upfront but made the whole thing a success.", "positive", "scaling", "skills"),
    ("The learning curve was steep but our team got there. Now AI is second nature and adoption keeps growing organically.", "positive", "scaling", "skills"),
    ("We paired the rollout with structured training and it made a huge difference. People who were sceptical are now evangelists.", "positive", "scaling", "skills"),
    ("Building internal AI literacy took time but was worth every hour. Now the team spots opportunities we'd never have seen.", "positive", "scaling", "skills"),
    ("We didn't underestimate the skills investment. Because we prepared properly, the scaling has gone really smoothly.", "positive", "scaling", "skills"),
    ("Our people needed upskilling before we could scale, and we invested in that. Happy we did — the capability is now a real asset.", "positive", "scaling", "skills"),

    # ── POSITIVE / PILOTING / NONE ────────────────────────────────────────────
    ("Our AI pilot for customer service has been brilliant — response times halved and satisfaction scores are up.", "positive", "piloting", "none"),
    ("Three months into our AI pilot and results are better than we projected. Full rollout planned for next quarter.", "positive", "piloting", "none"),
    ("We've been testing an AI scheduling tool and the efficiency gains are clear. The team is enthusiastic about expanding it.", "positive", "piloting", "none"),
    ("Our pilot with AI invoicing has been a revelation — errors down, admin time cut, team much happier.", "positive", "piloting", "none"),
    ("The AI chatbot pilot has exceeded expectations. Customers get faster answers and our staff focus on complex issues.", "positive", "piloting", "none"),
    ("Six weeks into an AI analytics pilot and we've already uncovered insights that changed how we think about our market.", "positive", "piloting", "none"),
    ("Piloting an AI content tool and it's been genuinely impressive. Our marketing output has doubled in quality and speed.", "positive", "piloting", "none"),
    ("The pilot is going really well. It's early days but the business case is already clear — we'll be rolling it out fully.", "positive", "piloting", "none"),

    # ── POSITIVE / PILOTING / SKILLS ──────────────────────────────────────────
    ("Our pilot needed more training than expected but the results make it worthwhile. Team is getting there and results are promising.", "positive", "piloting", "skills"),
    ("Skills took time to build during the pilot but we got there. Now the team actually enjoys using the AI tools.", "positive", "piloting", "skills"),
    ("We paired the pilot with a training programme and the combination worked well. Strong results, capable team.", "positive", "piloting", "skills"),
    ("The learning curve was steeper than the vendor suggested but the outcomes are positive enough that we're pushing through.", "positive", "piloting", "skills"),
    ("We underestimated how much training the pilot would need, but we committed to it and the results have justified that.", "positive", "piloting", "skills"),
    ("Our team was a bit underprepared technically when we started the pilot, but they've grown into it and the tool is delivering.", "positive", "piloting", "skills"),
    ("There's still a skills gap on the team but the pilot results are strong enough that we're investing in training to close it.", "positive", "piloting", "skills"),
    ("The pilot requires more hands-on support than I expected, but the value is real. We're upskilling the team to keep momentum.", "positive", "piloting", "skills"),

    # ── POSITIVE / EXPLORING / NONE ───────────────────────────────────────────
    ("I've been researching AI solutions for months and I'm genuinely excited. The right tool for our size definitely exists.", "positive", "exploring", "none"),
    ("We attended an AI summit last month and came away energised. We're actively evaluating tools and demos lined up.", "positive", "exploring", "none"),
    ("The more I explore AI options, the more convinced I am it's the right move. Something will fit our use case.", "positive", "exploring", "none"),
    ("We're in research mode but very optimistic. AI feels like a genuine opportunity for a business our size.", "positive", "exploring", "none"),
    ("Exciting time to be exploring AI. We've shortlisted three vendors and are doing demos next month.", "positive", "exploring", "none"),
    ("We haven't committed yet but everything we're seeing makes us want to move faster on this.", "positive", "exploring", "none"),
    ("Just starting to explore AI but the potential is clear. Taking our time to find the right fit.", "positive", "exploring", "none"),
    ("At the research stage but very enthusiastic. AI could help us genuinely compete with much larger players.", "positive", "exploring", "none"),

    # ── POSITIVE / EXPLORING / COST ───────────────────────────────────────────
    ("We're exploring AI tools seriously but keeping a close eye on cost. Want to find best value before committing.", "positive", "exploring", "cost"),
    ("Interested in AI and actively looking — just need to find an option that fits our budget without sacrificing quality.", "positive", "exploring", "cost"),
    ("We're optimistic about AI but want to be smart about the investment. Exploring options to find the most affordable entry point.", "positive", "exploring", "cost"),
    ("Really keen on AI adoption, but we need the pricing to make sense for a business our size. Still looking.", "positive", "exploring", "cost"),
    ("AI is clearly the right direction for us. Just need to find a cost model that works — exploring options actively.", "positive", "exploring", "cost"),
    ("We're enthusiastic about AI but won't rush into something that breaks the budget. Taking time to find the right deal.", "positive", "exploring", "cost"),
    ("Very positive on AI. The main thing holding us to the research phase is finding the right price point for our scale.", "positive", "exploring", "cost"),
    ("We want to adopt AI and are actively exploring. Cost is a factor but we're confident there's a sensible option out there.", "positive", "exploring", "cost"),

    # ── NEUTRAL / EXPLORING / COST ────────────────────────────────────────────
    ("We've been looking at AI tools but every solution we find is more expensive than expected. Still in research phase.", "neutral", "exploring", "cost"),
    ("AI interests us but pricing models are confusing and often don't match what's advertised. Cautiously exploring.", "neutral", "exploring", "cost"),
    ("Not opposed to AI but the total cost of ownership isn't clear. Not ready to commit until we understand the numbers.", "neutral", "exploring", "cost"),
    ("Exploring AI options but every demo leads to a pricing conversation we're not ready for. Taking our time.", "neutral", "exploring", "cost"),
    ("On the fence. The tools look useful but we'd need clearer ROI data before spending that kind of money.", "neutral", "exploring", "cost"),
    ("AI is appealing but we're a lean team and every cost decision matters. Still searching for something that makes financial sense.", "neutral", "exploring", "cost"),
    ("We've shortlisted a few AI tools but the price points are giving us pause. Holding off until we have more clarity.", "neutral", "exploring", "cost"),
    ("Interested but cautious. The tools we've seen are promising; the price tags less so. Still evaluating.", "neutral", "exploring", "cost"),

    # ── NEUTRAL / EXPLORING / SKILLS ──────────────────────────────────────────
    ("We're interested in AI but honestly not sure our team has the technical background to implement it without a lot of support.", "neutral", "exploring", "skills"),
    ("Exploring AI but the skill requirements feel steep. We'd need either training or external help — neither is cheap or quick.", "neutral", "exploring", "skills"),
    ("AI is on the agenda but our team is not particularly technical. Trying to understand what support we'd need.", "neutral", "exploring", "skills"),
    ("We like the idea of AI but the setup and management looks complex. Not sure we have the internal expertise.", "neutral", "exploring", "skills"),
    ("Looking at AI options but concerned about the learning curve. Our team are great at their jobs but aren't tech-savvy.", "neutral", "exploring", "skills"),
    ("AI could help us but we'd need significant training first. Exploring cautiously — don't want to set ourselves up to fail.", "neutral", "exploring", "skills"),
    ("In research mode but increasingly aware that our skills gap is a real barrier. Not sure how to bridge it affordably.", "neutral", "exploring", "skills"),
    ("Interested in AI but feel we'd need to hire someone technical to make it work. That changes the cost-benefit calculation.", "neutral", "exploring", "skills"),

    # ── NEUTRAL / EXPLORING / TRUST ───────────────────────────────────────────
    ("We're exploring AI but the more we read about errors and biases, the more cautious we feel. Not ready to commit.", "neutral", "exploring", "trust"),
    ("AI is interesting but reliability worries us. We've seen too many examples of AI getting it wrong in ways that matter.", "neutral", "exploring", "trust"),
    ("Looking into AI but data privacy is a big concern. We handle customer data and can't afford to get it wrong.", "neutral", "exploring", "trust"),
    ("We're researching AI tools but haven't found one we're confident enough to trust with real business decisions yet.", "neutral", "exploring", "trust"),
    ("Cautiously exploring AI. Some tools look promising but we need more evidence of reliability before using them with customers.", "neutral", "exploring", "trust"),
    ("On the fence about AI. The technology seems impressive but we've had one bad experience with automation before.", "neutral", "exploring", "trust"),
    ("We're looking at AI options but the black-box nature of some tools makes us uncomfortable. We want to understand what it does.", "neutral", "exploring", "trust"),
    ("Interested in AI but worried about vendor lock-in and what happens to our data. Exploring very carefully.", "neutral", "exploring", "trust"),

    # ── NEUTRAL / PILOTING / COST ─────────────────────────────────────────────
    ("We're running an AI pilot but ongoing costs are higher than forecast. Useful, but not sure the numbers stack up.", "neutral", "piloting", "cost"),
    ("Our pilot has shown some value but per-user licensing is expensive at scale. Waiting to negotiate better rates.", "neutral", "piloting", "cost"),
    ("Testing AI currently — it does what it says but the cost per use case is hard to justify for a business our size.", "neutral", "piloting", "cost"),
    ("Pilot is underway. Early signs are mixed — the tool works but the price point feels high given the efficiency gains so far.", "neutral", "piloting", "cost"),
    ("Running a pilot now. The AI does what we need but when I work out the annual cost, the ROI is marginal at best.", "neutral", "piloting", "cost"),
    ("We're testing it and it's fine, but fine isn't worth the price tag. Need to see more compelling results before committing.", "neutral", "piloting", "cost"),
    ("Pilot is going okay but every vendor conversation circles back to upselling. Cost is becoming a sticking point.", "neutral", "piloting", "cost"),
    ("We're in the pilot phase and the tool has genuine uses, but the subscription model at full scale worries us.", "neutral", "piloting", "cost"),

    # ── NEUTRAL / PILOTING / SKILLS ───────────────────────────────────────────
    ("Our AI pilot is in progress but adoption has been patchy. Some team members use it well, others are struggling.", "neutral", "piloting", "skills"),
    ("We're piloting an AI tool but we've hit a learning curve. It's not as intuitive as the vendor suggested.", "neutral", "piloting", "skills"),
    ("Testing AI now. The tool itself is capable but our team needs more hand-holding than we anticipated.", "neutral", "piloting", "skills"),
    ("Pilot is live but momentum is slow. The AI works but half the team doesn't feel confident enough to use it independently.", "neutral", "piloting", "skills"),
    ("We're running a pilot and results are mixed. The staff who get it see real value; the others find it frustrating.", "neutral", "piloting", "skills"),
    ("AI pilot has started but we underestimated the training investment. Spending more time on internal support than expected.", "neutral", "piloting", "skills"),
    ("Testing an AI tool and it shows promise, but we need a proper training programme before we can roll it out fully.", "neutral", "piloting", "skills"),
    ("Pilot is running but the skills gap in the team is holding back adoption. Useful tool, underwhelming uptake so far.", "neutral", "piloting", "skills"),

    # ── NEUTRAL / PILOTING / TRUST ────────────────────────────────────────────
    ("We're piloting an AI tool but outputs aren't always reliable enough for our standards. Still evaluating.", "neutral", "piloting", "trust"),
    ("Our pilot is running but we've had a few cases where the AI gave wrong information to customers. Reviewing carefully.", "neutral", "piloting", "trust"),
    ("Testing AI at the moment. It mostly works but there are edge cases where the output is just wrong. Keeping humans in the loop.", "neutral", "piloting", "trust"),
    ("Pilot is live but we're not ready to trust it without human oversight. Some outputs have been inconsistent.", "neutral", "piloting", "trust"),
    ("Running a pilot and the AI is capable but we've found some accuracy issues that are giving us pause.", "neutral", "piloting", "trust"),
    ("We're testing an AI tool but data handling practices aren't entirely clear to us. Waiting for legal to sign off.", "neutral", "piloting", "trust"),
    ("The pilot is useful in controlled conditions but we've had enough unexpected outputs to be cautious about full deployment.", "neutral", "piloting", "trust"),
    ("We're piloting carefully and keeping a human check on all outputs. The tool isn't reliable enough yet to operate unsupervised.", "neutral", "piloting", "trust"),

    # ── NEUTRAL / SCALING / COST ──────────────────────────────────────────────
    ("We've scaled AI across the business but the cumulative cost is now significant. Still valuable but watching the budget closely.", "neutral", "scaling", "cost"),
    ("AI is deployed widely but we've had to renegotiate contracts twice because costs kept creeping up.", "neutral", "scaling", "cost"),
    ("We've rolled out AI broadly but the pricing model changed after year one. Reassessing whether to continue at this scale.", "neutral", "scaling", "cost"),
    ("AI is live across most departments but we're concerned about sustainability. The value is real but so is the monthly spend.", "neutral", "scaling", "cost"),
    ("We've scaled but the cost base has grown with it. Worth it so far, but we're monitoring closely.", "neutral", "scaling", "cost"),
    ("AI is part of our stack now but the total spend is higher than we'd like. Looking for ways to optimise without losing capability.", "neutral", "scaling", "cost"),
    ("We've expanded AI use significantly but need to be more disciplined about which use cases actually justify the cost.", "neutral", "scaling", "cost"),
    ("Scaling AI has been broadly positive but the financial side needs more active management than we anticipated.", "neutral", "scaling", "cost"),

    # ── NEUTRAL / SCALING / SKILLS ────────────────────────────────────────────
    ("AI is deployed across the business but usage is inconsistent. Skills vary a lot between departments.", "neutral", "scaling", "skills"),
    ("We've scaled AI but we're battling an adoption gap — some teams use it brilliantly, others barely touch it.", "neutral", "scaling", "skills"),
    ("AI is live everywhere but we're struggling to maintain the skills needed to keep it running well. High turnover doesn't help.", "neutral", "scaling", "skills"),
    ("We've expanded AI use significantly but training is an ongoing burden. It's worth it, but it's a real commitment.", "neutral", "scaling", "skills"),
    ("AI is scaled but the capability isn't evenly distributed. Some staff are excellent, others need significant ongoing support.", "neutral", "scaling", "skills"),
    ("We've rolled out AI but the skills investment has been higher than budgeted. The value is there but so is the overhead.", "neutral", "scaling", "skills"),
    ("Scaling went okay but we underestimated how much ongoing training would be needed as the tools update and evolve.", "neutral", "scaling", "skills"),
    ("AI is in use across the business but the skills gap is a persistent issue. We're managing but it takes resource.", "neutral", "scaling", "skills"),

    # ── NEUTRAL / SCALING / TRUST ─────────────────────────────────────────────
    ("AI is deployed but we've had reliability incidents that have made us cautious. Still using it but with more oversight.", "neutral", "scaling", "trust"),
    ("We've scaled AI use but recent accuracy issues have made the team lose some confidence in the outputs.", "neutral", "scaling", "trust"),
    ("AI is embedded in our ops but there have been a few customer-facing errors that prompted us to review our governance.", "neutral", "scaling", "trust"),
    ("We're running AI at scale but keeping close human oversight after some unexpected outputs earlier in the year.", "neutral", "scaling", "trust"),
    ("AI is live but trust is fragile after two incidents last quarter. We've tightened controls and are monitoring closely.", "neutral", "scaling", "trust"),
    ("We've scaled successfully but remain guarded. The outputs are usually good; it's the exceptions that concern us.", "neutral", "scaling", "trust"),
    ("Running AI at scale but data governance has become a bigger issue than we anticipated. Working through it.", "neutral", "scaling", "trust"),
    ("AI is deployed widely but we've introduced human-in-the-loop checks after some reliability concerns. Cautiously optimistic.", "neutral", "scaling", "trust"),

    # ── NEUTRAL / NOT_INTERESTED / COST ───────────────────────────────────────
    ("We've looked at AI and the costs just don't make sense for us right now. Not ruling it out long term, but not now.", "neutral", "not_interested", "cost"),
    ("AI isn't on our roadmap currently — the pricing models we've seen don't fit our budget or scale.", "neutral", "not_interested", "cost"),
    ("We've done the analysis and can't justify the spend at this point. Maybe when we're bigger or prices come down.", "neutral", "not_interested", "cost"),
    ("Not pursuing AI for now. It's not that we're against it — the cost-benefit just doesn't add up at our current size.", "neutral", "not_interested", "cost"),
    ("We evaluated AI options and put it on hold. The financial case isn't there yet for a business at our stage.", "neutral", "not_interested", "cost"),
    ("AI is interesting but not a priority right now given the costs involved. We'll revisit when conditions change.", "neutral", "not_interested", "cost"),
    ("We're not moving forward with AI at the moment — costs are prohibitive relative to the value we'd get at our scale.", "neutral", "not_interested", "cost"),
    ("AI adoption is on ice for now. The pricing structure doesn't work for us, though we're open to revisiting in future.", "neutral", "not_interested", "cost"),

    # ── NEUTRAL / NOT_INTERESTED / SKILLS ─────────────────────────────────────
    ("We're not pursuing AI right now — we don't have the skills internally and it's not the right time to hire for them.", "neutral", "not_interested", "skills"),
    ("AI is not on our near-term roadmap. The capability gap is too large to bridge without significant investment in people.", "neutral", "not_interested", "skills"),
    ("We've decided to hold off on AI. We'd need to build or buy skills we don't have, and now isn't the right time.", "neutral", "not_interested", "skills"),
    ("Not moving forward with AI for now. We're a small team and don't have the technical depth to make it work properly.", "neutral", "not_interested", "skills"),
    ("AI adoption is paused. The skills requirements are real and we're not in a position to address them right now.", "neutral", "not_interested", "skills"),
    ("We looked at AI and decided the skills investment was too large for us at this stage. Revisiting next year maybe.", "neutral", "not_interested", "skills"),
    ("We're not pursuing AI currently — getting the skills in place would require hiring or retraining we can't prioritise.", "neutral", "not_interested", "skills"),
    ("AI isn't happening for us right now. The technical requirements exceed what our current team can handle.", "neutral", "not_interested", "skills"),

    # ── NEUTRAL / NOT_INTERESTED / TRUST ──────────────────────────────────────
    ("We're holding off on AI for now. We want to see more evidence of reliability before we'd be comfortable adopting it.", "neutral", "not_interested", "trust"),
    ("AI is not something we're pursuing currently — we're not confident enough in the outputs to use it in customer-facing processes.", "neutral", "not_interested", "trust"),
    ("We've decided to wait on AI. The technology is evolving fast and we'd rather let it mature before committing.", "neutral", "not_interested", "trust"),
    ("Not moving ahead with AI at this stage. We want more proven track record before trusting it with business-critical tasks.", "neutral", "not_interested", "trust"),
    ("We're not interested in AI right now — data privacy regulations are still catching up with the technology and we're cautious.", "neutral", "not_interested", "trust"),
    ("AI adoption is on hold pending better evidence of accuracy and reliability. We'll reassess when those questions are answered.", "neutral", "not_interested", "trust"),
    ("We're open to AI in principle but not ready to adopt it yet. Need to see stronger reliability evidence first.", "neutral", "not_interested", "trust"),
    ("Not pursuing AI currently — more interested in seeing how the regulatory and reliability landscape develops first.", "neutral", "not_interested", "trust"),

    # ── NEGATIVE / NOT_INTERESTED / COST ──────────────────────────────────────
    ("We looked at AI seriously and the costs are just not feasible for a business our size. Not pursuing this.", "negative", "not_interested", "cost"),
    ("The price tags for AI tools are completely out of range for us. We've parked the idea indefinitely.", "negative", "not_interested", "cost"),
    ("Every AI solution we evaluated required investment we simply don't have. It's not for us.", "negative", "not_interested", "cost"),
    ("The numbers don't add up. Too expensive for the value it would deliver at our scale. We've moved on.", "negative", "not_interested", "cost"),
    ("I looked into AI for our business and the setup costs alone would take years to recoup. We've decided against it.", "negative", "not_interested", "cost"),
    ("AI is priced for larger businesses. For a team of twelve like ours it's not economically viable.", "negative", "not_interested", "cost"),
    ("We explored it and walked away. Licensing, integration, and ongoing costs are too much for where we are financially.", "negative", "not_interested", "cost"),
    ("Not interested in AI. The ROI projections didn't stack up and we can't justify the capital expenditure.", "negative", "not_interested", "cost"),

    # ── NEGATIVE / NOT_INTERESTED / SKILLS ────────────────────────────────────
    ("We'd need to completely retrain our team or hire specialists to implement AI. That's not realistic for us.", "negative", "not_interested", "skills"),
    ("Our team doesn't have the technical background for AI adoption and we can't afford to build that capability. Moving on.", "negative", "not_interested", "skills"),
    ("AI requires expertise we don't have and can't easily acquire. Not viable without significant hiring.", "negative", "not_interested", "skills"),
    ("We looked at AI but the skills gap in our organisation makes proper implementation impossible. Not the right time.", "negative", "not_interested", "skills"),
    ("I tried to get buy-in for AI but the honest reality is our team isn't equipped for it. We've decided not to pursue it.", "negative", "not_interested", "skills"),
    ("No interest in AI adoption right now. We don't have the in-house technical knowledge and outsourcing it feels risky.", "negative", "not_interested", "skills"),
    ("The implementation complexity requires skills we don't have. We decided it wasn't worth the disruption.", "negative", "not_interested", "skills"),
    ("We considered AI but between the technical setup and staff training, the human cost ruled it out for us.", "negative", "not_interested", "skills"),

    # ── NEGATIVE / NOT_INTERESTED / TRUST ─────────────────────────────────────
    ("We decided against AI. The data privacy implications are too significant for us to be comfortable.", "negative", "not_interested", "trust"),
    ("I don't trust AI to interact with our clients. One bad output could damage relationships we've spent years building.", "negative", "not_interested", "trust"),
    ("Not going down the AI route. Too many stories of errors and bias. We'd rather keep humans in control.", "negative", "not_interested", "trust"),
    ("We looked at AI and decided the reliability risks were too high. Not something we're willing to gamble on.", "negative", "not_interested", "trust"),
    ("Our customers expect accuracy and consistency. AI can't guarantee that. Sticking with our existing processes.", "negative", "not_interested", "trust"),
    ("Data security concerns killed our AI interest. We can't have customer data going through third-party AI systems.", "negative", "not_interested", "trust"),
    ("Not interested in AI. We had one experience with a recommendation system that gave dangerous outputs. Not repeating that.", "negative", "not_interested", "trust"),
    ("We evaluated several AI tools and came away more worried than when we started. The trust just isn't there for us.", "negative", "not_interested", "trust"),

    # ── NEGATIVE / EXPLORING / COST ───────────────────────────────────────────
    ("We started looking into AI but stopped when we saw the pricing. Just not accessible for small businesses like ours.", "negative", "exploring", "cost"),
    ("Did a bit of research on AI tools and immediately hit a wall — costs are far beyond what we can justify.", "negative", "exploring", "cost"),
    ("We tried to explore AI options but every path led to a price that stopped us cold. It's for bigger companies.", "negative", "exploring", "cost"),
    ("Began looking into AI last quarter and pulled back fast. The cost of entry is too high for our margin structure.", "negative", "exploring", "cost"),
    ("We explored AI briefly but the financial reality killed the idea quickly. The industry hasn't made this accessible yet.", "negative", "exploring", "cost"),
    ("Looked at a few platforms. The first quotes we got were so far out of range we stopped exploring entirely.", "negative", "exploring", "cost"),
    ("We were keen to explore AI but the pricing completely put us off. There's a real gap in the market for affordable SME tools.", "negative", "exploring", "cost"),
    ("Tried to research AI options. Every credible solution was priced for enterprise, not small businesses like us.", "negative", "exploring", "cost"),

    # ── NEGATIVE / EXPLORING / SKILLS ─────────────────────────────────────────
    ("We started exploring AI but quickly realised our team doesn't have the capability to implement or maintain it.", "negative", "exploring", "skills"),
    ("Did some initial research on AI but the technical requirements are way beyond our current team's abilities.", "negative", "exploring", "skills"),
    ("Explored AI options but concluded that without dedicated technical staff, we'd be setting ourselves up to fail.", "negative", "exploring", "skills"),
    ("We tried to research AI tools but every solution assumed a level of technical expertise we don't have. Gave up.", "negative", "exploring", "skills"),
    ("Started down the AI exploration path but the skills barrier was obvious very quickly. We don't have what's needed.", "negative", "exploring", "skills"),
    ("We looked into AI but each product we evaluated required specialist knowledge to set up and run. Not feasible for us.", "negative", "exploring", "skills"),
    ("AI exploration was discouraging. The tools expect you to have a data science background just to get started.", "negative", "exploring", "skills"),
    ("We explored AI seriously but the skills required to implement it properly aren't something we can realistically build.", "negative", "exploring", "skills"),

    # ── NEGATIVE / EXPLORING / TRUST ──────────────────────────────────────────
    ("We looked into AI but the more we learned about how it works, the less comfortable we felt. Too many unknowns.", "negative", "exploring", "trust"),
    ("Started exploring AI and quickly became concerned about what vendors do with our data. Decided it wasn't worth it.", "negative", "exploring", "trust"),
    ("We did some research on AI tools and came away deeply sceptical. The accuracy claims don't match what we've heard.", "negative", "exploring", "trust"),
    ("Explored AI briefly and hit too many red flags around data practices and model reliability. Not for us.", "negative", "exploring", "trust"),
    ("After reading the small print on data usage, we decided we couldn't trust these companies with our business information.", "negative", "exploring", "trust"),
    ("We started exploring AI tools but the liability and accuracy questions were never answered satisfactorily. We stopped.", "negative", "exploring", "trust"),
    ("Research into AI made us more sceptical, not less. We couldn't find a tool we'd be comfortable trusting with real decisions.", "negative", "exploring", "trust"),
    ("We explored AI options but the combination of data privacy concerns and inconsistent accuracy made it a non-starter.", "negative", "exploring", "trust"),

    # ── NEGATIVE / PILOTING / COST ────────────────────────────────────────────
    ("We ran an AI pilot and ended it early. Cost per result was far too high to justify at any scale.", "negative", "piloting", "cost"),
    ("Our pilot was a disappointment — not because the AI was terrible, but because the pricing model was unworkable for us.", "negative", "piloting", "cost"),
    ("We trialled an AI tool for two months and terminated it. The ongoing costs weren't sustainable.", "negative", "piloting", "cost"),
    ("Our AI pilot showed some promise but costs escalated quickly. We couldn't see a path to affordability.", "negative", "piloting", "cost"),
    ("We piloted an AI solution and while it had some value, the pricing structure meant we couldn't scale it. Ended the contract.", "negative", "piloting", "cost"),
    ("The pilot confirmed what we suspected — the AI works but it's too expensive for a business our size to sustain.", "negative", "piloting", "cost"),
    ("We ran a pilot and the technology was fine, but the cost-per-outcome made no business sense. We pulled out.", "negative", "piloting", "cost"),
    ("Our pilot gave us a clear answer: the tool isn't affordable at the usage level we'd need to make it worthwhile.", "negative", "piloting", "cost"),

    # ── NEGATIVE / PILOTING / SKILLS ──────────────────────────────────────────
    ("We ran a pilot but it failed. Our team couldn't get to grips with the tool despite weeks of training.", "negative", "piloting", "skills"),
    ("Our AI pilot was unsuccessful. The technology needed more technical management than we could provide.", "negative", "piloting", "skills"),
    ("We tried an AI pilot and abandoned it. The skill requirements to run it properly were too demanding for our team.", "negative", "piloting", "skills"),
    ("Ran a pilot for three months — it just didn't work for us. The interface was complex and the team never got comfortable.", "negative", "piloting", "skills"),
    ("Our AI pilot didn't work out. Every time something broke, we needed external support. Not sustainable.", "negative", "piloting", "skills"),
    ("The pilot failed because we simply couldn't operate the tool without constant help from the vendor. That's not workable.", "negative", "piloting", "skills"),
    ("We tried an AI pilot but the knowledge gap in our team was too large. Despite real effort, adoption just didn't happen.", "negative", "piloting", "skills"),
    ("Our pilot ended because the tool requires specialist skills to use properly, and we don't have them and can't hire them.", "negative", "piloting", "skills"),

    # ── NEGATIVE / PILOTING / TRUST ───────────────────────────────────────────
    ("We piloted an AI tool and found outputs unreliable. It gave wrong information to customers on multiple occasions.", "negative", "piloting", "trust"),
    ("Our AI pilot ended badly. We had two incidents where the AI made decisions we would never have approved. Trust destroyed.", "negative", "piloting", "trust"),
    ("We tested an AI system but it kept producing factually incorrect outputs. Had to pull the plug.", "negative", "piloting", "trust"),
    ("We ran a pilot and stopped it. The AI was confident but often wrong. That combination is dangerous in our line of work.", "negative", "piloting", "trust"),
    ("Our pilot revealed serious issues with accuracy and data handling. We ended it and have no plans to return to AI.", "negative", "piloting", "trust"),
    ("The pilot showed the AI couldn't be trusted with unsupervised decisions. Too many errors in real-world conditions.", "negative", "piloting", "trust"),
    ("We ran an AI pilot that we had to terminate after the tool gave a customer significantly wrong information. Trust gone.", "negative", "piloting", "trust"),
    ("The pilot was a wake-up call. The AI performed well on demos but fell apart on real data. Not safe to use.", "negative", "piloting", "trust"),
]


def main():
    os.makedirs("data", exist_ok=True)

    combos = list({(s, st, b) for _, s, st, b in TEMPLATES})
    combo_templates = {c: [t for t in TEMPLATES if (t[1], t[2], t[3]) == c] for c in combos}

    rows = []
    for _ in range(500):
        combo = random.choice(combos)
        text, sentiment, stage, barrier = random.choice(combo_templates[combo])
        rows.append((text, sentiment, stage, barrier))

    with open("data/sme_survey_responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["response", "sentiment", "adoption_stage", "main_barrier"])
        for text, sentiment, stage, barrier in rows:
            writer.writerow([text, sentiment, stage, barrier])

    print(f"Generated {len(rows)} rows -> data/sme_survey_responses.csv")
    print(f"Templates: {len(TEMPLATES)} unique | Combinations: {len(combos)}")
    print(f"\nClass distribution:")
    print(f"  sentiment:      {dict(sorted(Counter(r[1] for r in rows).items()))}")
    print(f"  adoption_stage: {dict(sorted(Counter(r[2] for r in rows).items()))}")
    print(f"  main_barrier:   {dict(sorted(Counter(r[3] for r in rows).items()))}")


if __name__ == "__main__":
    main()
