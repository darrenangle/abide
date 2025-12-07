"""
Public domain poems and synthetic examples for form validation.

All real-world poems are public domain (pre-1928 or explicitly PD).
Synthetic poems are designed to pass 100% for framework validation.
"""

# =============================================================================
# HAIKU (5-7-5 syllables, 3 lines)
# =============================================================================

HAIKU_BASHO_OLD_POND = """\
An old silent pond
A frog jumps into the pond
Splash! Silence again"""
# Matsuo Basho (1644-1694), translated - public domain

HAIKU_SYNTHETIC_PERFECT = """\
The morning sun glows
Cherry blossoms gently fall
Spring has come at last"""
# Synthetic: exactly 5-7-5 syllables

HAIKU_BUSON_SPRING_SEA = """\
The spring sea rising
And falling, rising and then
Falling all day long"""
# Yosa Buson (1716-1784) - public domain


# =============================================================================
# TANKA (5-7-5-7-7 syllables, 5 lines)
# =============================================================================

TANKA_SYNTHETIC_PERFECT = """\
The autumn moon shines
Casting silver on the lake
Gentle ripples spread
While the night birds call softly
Dreams drift on the quiet waves"""
# Synthetic: exactly 5-7-5-7-7 syllables


# =============================================================================
# LIMERICK (5 lines, AABBA rhyme scheme)
# =============================================================================

LIMERICK_LEAR_OLD_MAN_BEARD = """\
There was an Old Man with a beard,
Who said, It is just as I feared!
Two Owls and a Hen,
Four Larks and a Wren,
Have all built their nests in my beard!"""
# Edward Lear (1812-1888), "A Book of Nonsense" (1846) - public domain

LIMERICK_SYNTHETIC_PERFECT = """\
A writer who lived by the sea
Wrote verses with patience and glee
He crafted each line
To make the words shine
And shared them for all who could see"""
# Synthetic: AABBA rhyme, 5 lines


# =============================================================================
# SHAKESPEAREAN SONNET (14 lines, ABAB CDCD EFEF GG, iambic pentameter)
# =============================================================================

SONNET_SHAKESPEARE_18 = """\
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee."""
# William Shakespeare, Sonnet 18 (1609) - public domain

SONNET_SHAKESPEARE_130 = """\
My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.
I have seen roses damasked, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet well I know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground.
And yet, by heaven, I think my love as rare
As any she belied with false compare."""
# William Shakespeare, Sonnet 130 (1609) - public domain


# =============================================================================
# PETRARCHAN SONNET (14 lines, ABBAABBA CDECDE or CDCDCD)
# =============================================================================

SONNET_MILTON_BLINDNESS = """\
When I consider how my light is spent,
Ere half my days, in this dark world and wide,
And that one Talent which is death to hide
Lodged with me useless, though my Soul more bent
To serve therewith my Maker, and present
My true account, lest he returning chide;
Doth God exact day-labour, light denied?
I fondly ask. But patience, to prevent
That murmur, soon replies, God doth not need
Either man's work or his own gifts; who best
Bear his mild yoke, they serve him best. His state
Is Kingly. Thousands at his bidding speed
And post o'er Land and Ocean without rest:
They also serve who only stand and wait."""
# John Milton, "On His Blindness" (c. 1655) - public domain


# =============================================================================
# VILLANELLE (19 lines, 5 tercets + quatrain, ABA rhyme, two refrains)
# =============================================================================

VILLANELLE_SYNTHETIC_PERFECT = """\
The stars will shine when day is done
And shadows fall across the land
We walk beneath the fading sun

The birds that sing have now begun
To nest where gentle breezes fanned
The stars will shine when day is done

The rivers flow and waters run
Through valleys by the master planned
We walk beneath the fading sun

What once was lost cannot be won
Yet still we reach with open hand
The stars will shine when day is done

Our journey here is never done
Though time slips by like grains of sand
We walk beneath the fading sun

When all our earthly work is spun
And we have made our final stand
The stars will shine when day is done
We walk beneath the fading sun"""
# Synthetic: 19 lines, ABA scheme
# A rhymes: done/sun/begun/run/won/done/spun/done/sun (all rhyme with -un sound)
# B rhymes: land/fanned/planned/hand/sand/stand (all rhyme with -and sound)
# Refrains: line 1 at 6,12,18 and line 3 at 9,15,19


# =============================================================================
# SESTINA (39 lines, 6 sestets + envoi, end-word rotation)
# =============================================================================

SESTINA_SYNTHETIC_PERFECT = """\
I walked alone beneath the ancient trees
And watched the light dance softly on the stream
The morning air was fresh with scent of flowers
While birds sang out their songs upon the breeze
I thought of days now faded into dreams
And felt the turning of the endless hours

I sat beside the bank for many hours
And listened to the whispers of the trees
The water flowed like silver through my dreams
A ribbon winding down the gentle stream
I closed my eyes and felt the cooling breeze
And breathed the fragrance of the morning flowers

The meadow there was carpeted with flowers
I wandered through them counting up the hours
The petals danced upon the summer breeze
Cast shadows underneath the swaying trees
I heard the distant murmur of the stream
And drifted slowly into waking dreams

How often I had walked here in my dreams
Through fields of gold and crimson colored flowers
Along the mossy edges of the stream
I passed away the long and languid hours
Beneath the sheltering branches of the trees
And felt upon my face the gentle breeze

There came a sudden stirring of the breeze
That woke me gently from my pleasant dreams
I saw the sunlight slanting through the trees
It fell like gold upon the waiting flowers
I knew that I had lingered here for hours
But still I sat beside the quiet stream

How beautiful and peaceful was that stream
How soft and sweet the ever-present breeze
I wished that I could stay for endless hours
And never wake from all my lovely dreams
Forever walking through the fragrant flowers
Forever resting underneath the trees

The stream flows on through dreams beneath the trees
While flowers bloom and hours pass with the breeze
And all my dreams drift softly through the hours"""
# Synthetic: 39 lines (6 sestets of 6 lines + 3-line envoi)
# End words: trees, stream, flowers, breeze, dreams, hours
# Envoi uses all 6 end words in 3 lines (2 per line)
# End words: trees, stream, flowers, breeze, dreams, hours


# =============================================================================
# TRIOLET (8 lines, ABaAabAB, lines 1=4=7, lines 2=8)
# =============================================================================

TRIOLET_HARDY_BIRDS = """\
How great my grief, my joys how few,
Since first it was my fate to know thee!
Have the slow years not brought to view
How great my grief, my joys how few,
Nor memory shaped old times anew,
Nor loving-Loss not brought me low? No,
How great my grief, my joys how few,
Since first it was my fate to know thee!"""
# Thomas Hardy (1840-1928) - public domain (died 1928)

TRIOLET_SYNTHETIC_PERFECT = """\
The roses bloom in early spring
And fill the air with sweet perfume
The morning birds begin to sing
The roses bloom in early spring
What joy and beauty they do bring
Before the summer's heat and gloom
The roses bloom in early spring
And fill the air with sweet perfume"""
# Synthetic: ABaAabAB, lines 1=4=7, lines 2=8


# =============================================================================
# RONDEAU (15 lines, AABBA AABR AABBAR, R=rentrement/refrain)
# =============================================================================

RONDEAU_MCCRAE_FLANDERS = """\
In Flanders fields the poppies blow
Between the crosses, row on row,
That mark our place; and in the sky
The larks, still bravely singing, fly
Scarce heard amid the guns below.

We are the Dead. Short days ago
We lived, felt dawn, saw sunset glow,
Loved and were loved, and now we lie
In Flanders fields.

Take up our quarrel with the foe:
To you from failing hands we throw
The torch; be yours to hold it high.
If ye break faith with us who die
We shall not sleep, though poppies grow
In Flanders fields."""
# John McCrae (1872-1918), "In Flanders Fields" (1915) - public domain


# =============================================================================
# PANTOUM (quatrains with line repetition, L2->L1 next, L4->L3 next)
# =============================================================================

PANTOUM_SYNTHETIC_PERFECT = """\
The rain falls soft upon the roof tonight
And shadows dance upon the window pane
I sit alone and read by candlelight
While thunder rumbles distantly again

And shadows dance upon the window pane
The storm moves slowly over hills afar
While thunder rumbles distantly again
I watch and wait beneath the evening star

The storm moves slowly over hills afar
The wind begins to whisper through the eaves
I watch and wait beneath the evening star
And listen to the rustling of the leaves

The wind begins to whisper through the eaves
The rain falls soft upon the roof tonight
And listen to the rustling of the leaves
I sit alone and read by candlelight"""
# Synthetic: 4 quatrains, L2->L1, L4->L3, circular ending


# =============================================================================
# TERZA RIMA (tercets with ABA BCB CDC chain rhyme)
# =============================================================================

TERZA_RIMA_SHELLEY_WEST_WIND_EXCERPT = """\
O wild West Wind, thou breath of Autumn's being,
Thou, from whose unseen presence the leaves dead
Are driven, like ghosts from an enchanter fleeing,

Yellow, and black, and pale, and hectic red,
Pestilence-stricken multitudes: O thou,
Who chariotest to their dark wintry bed

The winged seeds, where they lie cold and low,
Each like a corpse within its grave, until
Thine azure sister of the Spring shall blow"""
# Percy Bysshe Shelley, "Ode to the West Wind" (1819) - public domain
# First 3 tercets only

TERZA_RIMA_SYNTHETIC_PERFECT = """\
The morning light breaks golden on the hill
And wakes the sleeping valley down below
The birds begin their chorus sharp and shrill

Across the fields the gentle breezes blow
And carry scents of flowers on the air
The river sparkles in the morning glow

The trees stand tall and green beyond compare
Their branches reaching upward to the sky
The world awakes and life is everywhere

And so another day goes drifting by"""
# Synthetic: ABA BCB CDC D
# A rhymes: hill/shrill
# B rhymes: below/blow/glow
# C rhymes: air/compare/everywhere
# D rhymes: sky/by


# =============================================================================
# GHAZAL (5+ couplets, AA BA CA pattern with radif/refrain)
# =============================================================================

GHAZAL_SYNTHETIC_PERFECT = """\
The moon hangs low and bright tonight above
The stars shine clear and light tonight above

I walk alone through empty streets at dusk
My path is lit by sight tonight above

The wind has gone to sleep within the trees
The world is calm and right tonight above

What dreams may come to those who wait for dawn
When stars take flight tonight above

The poet writes these words with all his might
His verses take their flight tonight above"""
# Synthetic: 5 couplets, AA BA CA DA EA pattern
# Radif (refrain): "tonight above" - identical ending on all rhyming lines
# Qafiya (rhyme): bright/light/sight/right/flight/might/flight - all rhyme with "-ight" sound


# =============================================================================
# BALLADE (3 octaves + envoi, ABABBCBC per octave, refrain at end)
# =============================================================================

BALLADE_VILLON_LADIES_ROSSETTI = """\
Tell me now in what hidden way is
Lady Flora the lovely Roman?
Where's Hipparchia, and where is Thais,
Neither of them the fairer woman?
Where is Echo, beheld of no man,
Only heard on river and mere,
She whose beauty was more than human?
But where are the snows of yester-year?

Where's Heloise, the learned nun,
For whose sake Abeillard, I ween,
Lost manhood and put priesthood on?
(From Love he won such dule and teen!)
And where, I pray you, is the Queen
Who willed that Buridan should steer
Sewed in a sack's mouth down the Seine?
But where are the snows of yester-year?

White Queen Blanche, like a queen of lilies,
With a voice like any mermaiden,
Bertha Broadfoot, Beatrice, Alice,
And Ermengarde the lady of Maine,
And that good Joan whom Englishmen
At Rouen doomed and burned her there,
Mother of God, where are they then?
But where are the snows of yester-year?

Nay, never ask this week, fair lord,
Where they are gone, nor yet this year,
Except with this for an overword,
But where are the snows of yester-year?"""
# Francois Villon, "Ballade des Dames du Temps Jadis"
# Translated by Dante Gabriel Rossetti (1828-1882) - public domain


# =============================================================================
# CLERIHEW (4 lines, AABB, first line is a name)
# =============================================================================

CLERIHEW_BENTLEY_DAVY = """\
Sir Humphry Davy
Abominated gravy
He lived in the odium
Of having discovered sodium"""
# Edmund Clerihew Bentley (1875-1956), "Biography for Beginners" (1905) - public domain

CLERIHEW_SYNTHETIC_PERFECT = """\
Edgar Allan Poe
Wrote tales of woe and snow
His stories caused great fright
With ravens in the night"""
# Synthetic: AABB, first line is a name
# Poe/snow rhyme (1.0), fright/night rhyme (1.0)


# =============================================================================
# BLUES POEM (AAB tercets, L1 repeated/varied as L2)
# =============================================================================

BLUES_SYNTHETIC_PERFECT = """\
Woke up this morning with the sun in my eyes
Woke up this morning with the sun in my eyes
Looked out my window and I watched the blue birds rise

My baby left me standing at the door
My baby left me standing at the door
Now I don't know what I'm living for

The rain keeps falling down on this old street
The rain keeps falling down on this old street
And I got nothing left but my tired feet"""
# Synthetic: 3 AAB stanzas with L1 repeated exactly as L2
