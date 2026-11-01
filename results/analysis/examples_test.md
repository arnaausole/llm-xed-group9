# Test examples: QLoRA vs prompting

Made by analysis/find_examples.py.

## EN: qwen3-8b_qlora_en vs qwen3-8b_five-shot

Out of 508 lines: 111 improved, 76 worsened, 127 with exactly the same prediction.

### Improved by QLoRA (8 random examples)

- `e46a5edf11ac` 
  - EN: They're them, I'm still me.
  - ES: Ellos son ellos, yo sigo siendo yo.
  - gold: ['trust'] | prompting: ['sadness', 'trust'] | QLoRA: ['trust']
- `c64fb5c3ad0a` 
  - EN: I didn't believe in the prophecies.
  - ES: No creía en las profecías.
  - gold: ['disgust'] | prompting: ['disgust', 'sadness'] | QLoRA: ['disgust']
- `595a1a1fe3d5` 
  - EN: - He died kind of slow.
  - ES: - Murió lentamente.
  - gold: ['sadness'] | prompting: ['disgust', 'sadness'] | QLoRA: ['sadness']
- `e6371cff16ed` 
  - EN: There's no harm in hearing the young lady out, is there?
  - ES: No causa ningún daño oírla, ¿no?
  - gold: ['anticipation'] | prompting: ['surprise', 'trust'] | QLoRA: ['anticipation']
- `a2fb35e75c19` 
  - EN: Where is this manuscript?
  - ES: ¿Dónde está el manuscrito?
  - gold: ['anticipation'] | prompting: ['fear', 'sadness'] | QLoRA: ['anticipation']
- `3682796983cc` 
  - EN: - Listen, I'll pay his fare.
  - ES: - Yo pago su boleto.
  - gold: ['anticipation'] | prompting: ['trust'] | QLoRA: ['anger', 'anticipation']
- `9616c8a2bbc6` 
  - EN: Loomis, could we have two seconds alone?
  - ES: Loomis, ¿nos puedes dejar solos un momento?
  - gold: ['anticipation', 'trust'] | prompting: ['fear', 'sadness'] | QLoRA: ['anticipation']
- `d841116b0498` 
  - EN: - But what happens if...
  - ES: - Pero que pasa si...
  - gold: ['fear'] | prompting: ['anticipation', 'fear'] | QLoRA: ['fear']

### Worsened by QLoRA (8 random examples)

- `b06aeda061cb` 
  - EN: What was your relationship with the deceased?
  - ES: ¿Cuál era su relación con el fallecido?
  - gold: ['sadness'] | prompting: ['fear', 'sadness'] | QLoRA: ['anticipation']
- `6706df064fef` 
  - EN: Wouldn't you say that prison is my fate?
  - ES: Los dos fuimos sentenciados a cadena perpetua.
  - gold: ['fear', 'sadness'] | prompting: ['fear', 'sadness'] | QLoRA: ['sadness']
- `cfd00d8d76b9` 
  - EN: Don't forget me, Bastian.
  - ES: Bastián: no te olvides de mí.
  - gold: ['sadness', 'trust'] | prompting: ['sadness', 'trust'] | QLoRA: ['anticipation', 'joy', 'trust']
- `7225337cea67` 
  - EN: I can't, and God knows the department can't.
  - ES: Yo no puedo, y Dios sabe que el departamento tampoco puede.
  - gold: ['sadness'] | prompting: ['fear', 'sadness'] | QLoRA: ['anger', 'fear']
- `9e52ac15e263` 
  - EN: It'd be a big help if you could keep an area clear in front of the bridge for us.
  - ES: Podrían despejar un poco el área para bajar.
  - gold: ['trust'] | prompting: ['anticipation', 'trust'] | QLoRA: ['anticipation']
- `392dd1e59401` 
  - EN: You know when I realized that?
  - ES: ¿Sabes cuándo me di cuenta de eso?
  - gold: ['surprise'] | prompting: ['surprise'] | QLoRA: ['anticipation']
- `378ef6e71292` 
  - EN: I didn't spy on him
  - ES: Así es como lo hacía...
  - gold: ['trust'] | prompting: ['sadness', 'trust'] | QLoRA: ['anger', 'fear', 'sadness']
- `211a46e036e4` 
  - EN: Oh, you must be from around here.
  - ES: Ud. ha de ser de por aquí.
  - gold: ['surprise'] | prompting: ['surprise', 'trust'] | QLoRA: ['anticipation']

## ES: qwen3-8b_qlora_es vs qwen3-8b_five-shot

Out of 508 lines: 98 improved, 64 worsened, 122 with exactly the same prediction.

### Improved by QLoRA (8 random examples)

- `0e601d66cfd5` 
  - EN: - Listen, I'll pay his fare.
  - ES: - Yo pago su boleto. - Bueno.
  - gold: ['anticipation'] | prompting: ['trust'] | QLoRA: ['anger', 'anticipation']
- `26dc5c834024` 
  - EN: All those mothers had to make do with what nature gave them.
  - ES: Todas esas madres tienen que aceptar lo que les llega
  - gold: ['sadness'] | prompting: ['disgust', 'sadness'] | QLoRA: ['sadness']
- `61d660adef5d` 
  - EN: - Did you see the Giants?
  - ES: - ¿Viste a los gigantes?
  - gold: ['surprise'] | prompting: ['anticipation', 'surprise'] | QLoRA: ['surprise']
- `0fe8be6e9163` 
  - EN: I want one!
  - ES: , ¡yo quiero uno! .
  - gold: ['anticipation', 'surprise'] | prompting: ['joy'] | QLoRA: ['anticipation']
- `a7a51ea37d73` 
  - EN: Satan, stop!
  - ES: Satanás, para.
  - gold: ['anger'] | prompting: ['fear', 'surprise'] | QLoRA: ['anger']
- `074d4d8a0b67` 
  - EN: Yeah, that's it.
  - ES: Sí, ése es.
  - gold: ['joy'] | prompting: ['surprise'] | QLoRA: ['joy']
- `3a27d5398d46` 
  - EN: There's a real cab up there.
  - ES: Allá hay un taxi de verdad.
  - gold: ['joy'] | prompting: ['joy', 'surprise'] | QLoRA: ['joy']
- `502ae8f3e279` 
  - EN: It's gonna take some getting used to, isn't it?
  - ES: Te tomara algun tiempo acostumbrarte
  - gold: ['anticipation', 'fear'] | prompting: ['sadness'] | QLoRA: ['anticipation']

### Worsened by QLoRA (8 random examples)

- `c64fb5c3ad0a` 
  - EN: I didn't believe in the prophecies.
  - ES: No creía en las profecías.
  - gold: ['disgust'] | prompting: ['disgust', 'sadness'] | QLoRA: ['sadness']
- `7abd1e640f21` 
  - EN: - Grab the hand rail!
  - ES: - ¡Súbase en el techo!
  - gold: ['surprise'] | prompting: ['anticipation', 'surprise'] | QLoRA: ['anger']
- `ab2396d8029e` 
  - EN: You're a handsome and powerful man.
  - ES: Es usted muy guapo y un hombre de poder.
  - gold: ['trust'] | prompting: ['trust'] | QLoRA: ['joy', 'trust']
- `112bc03f4049` 
  - EN: You son of a bitch.
  - ES: Hijo de perra.
  - gold: ['anger', 'disgust'] | prompting: ['anger', 'disgust'] | QLoRA: ['anger']
- `1522b7785e88` 
  - EN: That's what always happens when people fall in love.
  - ES: Eso pasa cuando uno se enamora.
  - gold: ['sadness'] | prompting: ['fear', 'sadness'] | QLoRA: ['joy']
- `49a01bb12d53` 
  - EN: They tell me you're quite a stud player, young man.
  - ES: He oído que eres un buen jugador de stud póquer, joven.
  - gold: ['anticipation', 'trust'] | prompting: ['trust'] | QLoRA: ['joy']
- `daa314dc6b78` 
  - EN: Thank you.
  - ES: -Gracias.
  - gold: ['trust'] | prompting: ['trust'] | QLoRA: ['joy', 'trust']
- `8df8aab0009c` 
  - EN: Hey, hey, hey!
  - ES: - Eh, eh, eh.
  - gold: ['surprise'] | prompting: ['surprise'] | QLoRA: ['anticipation']

## ES: qwen3-8b_qlora_en vs qwen3-8b_five-shot

Out of 508 lines: 94 improved, 80 worsened, 107 with exactly the same prediction.

### Improved by QLoRA (8 random examples)

- `b7209e5bf2d6` 
  - EN: - What's the matter?
  - ES: ¡Maníacos! - ¿Qué pasa?
  - gold: ['anticipation'] | prompting: ['fear', 'surprise'] | QLoRA: ['anger', 'anticipation']
- `b4bef21d14df` 
  - EN: Oh, my God.
  - ES: ¡Dios mio!
  - gold: ['fear'] | prompting: ['fear', 'surprise'] | QLoRA: ['fear']
- `ad66c410ae86` 
  - EN: Watch it, folks!
  - ES: ¡Cuidado, amigos!
  - gold: ['fear'] | prompting: ['fear', 'surprise'] | QLoRA: ['fear']
- `0bd722512382` 
  - EN: He is inspired by the new openness here.
  - ES: Se siente inspirado por la apertura.
  - gold: ['joy'] | prompting: ['joy', 'surprise'] | QLoRA: ['joy']
- `a2fb35e75c19` 
  - EN: Where is this manuscript?
  - ES: ¿Dónde está el manuscrito?
  - gold: ['anticipation'] | prompting: ['fear', 'surprise'] | QLoRA: ['anticipation']
- `79ede7dc0453` 
  - EN: Wonderful!
  - ES: - Que dice, querido?
  - gold: ['anticipation'] | prompting: ['surprise'] | QLoRA: ['anticipation']
- `3c2e49653dc3` 
  - EN: Some people never learn.
  - ES: Algunas personas nunca aprenden
  - gold: ['anger'] | prompting: ['sadness'] | QLoRA: ['anger', 'disgust']
- `c64fb5c3ad0a` 
  - EN: I didn't believe in the prophecies.
  - ES: No creía en las profecías.
  - gold: ['disgust'] | prompting: ['disgust', 'sadness'] | QLoRA: ['disgust']

### Worsened by QLoRA (8 random examples)

- `7f3ebd08fb74` 
  - EN: - $10,000 for a taxi!
  - ES: - ¡Por favor! - ¡Le pagaremos $ 10000!
  - gold: ['surprise'] | prompting: ['anticipation', 'surprise'] | QLoRA: ['anger', 'anticipation']
- `2f7359e3e76e` 
  - EN: He's not making a whole lot of sense.
  - ES: No tiene mucho sentido.
  - gold: ['disgust'] | prompting: ['disgust'] | QLoRA: ['anger', 'disgust']
- `7631a67173ae` 
  - EN: Hey, that's it.
  - ES: ¡Tiempo!
  - gold: ['joy', 'surprise'] | prompting: ['surprise'] | QLoRA: ['anticipation']
- `e9c2e5b5c834` 
  - EN: It sometimes never goes away.
  - ES: La prensa está fuera, mamá.
  - gold: ['sadness'] | prompting: ['fear', 'sadness'] | QLoRA: ['fear']
- `392dd1e59401` 
  - EN: You know when I realized that?
  - ES: ¿Sabes cuándo me di cuenta de eso?
  - gold: ['surprise'] | prompting: ['sadness', 'surprise'] | QLoRA: ['anticipation']
- `e357496161cd` 
  - EN: I'm not ready to retire yet.
  - ES: No estoy listo para retirarme todavía.
  - gold: ['sadness'] | prompting: ['fear', 'sadness'] | QLoRA: ['anger', 'anticipation']
- `457dee85ad60` 
  - EN: Usually, I just reach out my hand, and obstacles disappear.
  - ES: Por lo general, estiro la mano y los obstáculos desaparecen.
  - gold: ['joy', 'surprise'] | prompting: ['joy', 'surprise'] | QLoRA: ['anticipation']
- `71d2ae510fc6` 
  - EN: We don't want Mario to come, the bald giant with stupid sunglasses, knocking everybody around for kicks.
  - ES: No queremos que ese gigante calvo de lentes oscuros sólo venga a pegarnos.
  - gold: ['disgust'] | prompting: ['disgust', 'fear'] | QLoRA: ['anger', 'fear']
