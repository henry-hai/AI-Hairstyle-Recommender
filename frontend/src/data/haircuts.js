// Maps a recommended style name (free-form text from the backend) to a
// reference photo in /public/haircuts. Matching is keyword-based and
// order-sensitive: the FIRST keyword found in the style name wins, so more
// specific cuts are listed before generic ones.

const STYLE_IMAGE_RULES = [
  { keyword: 'pompadour', image: 'pompadour' },
  { keyword: 'quiff', image: 'quiff' },
  { keyword: 'undercut', image: 'undercut' },
  { keyword: 'slick', image: 'slick-back' },
  { keyword: 'faux', image: 'faux-hawk' },
  { keyword: 'flat top', image: 'flat-top' },
  { keyword: 'crew', image: 'crew-cut' },
  { keyword: 'buzz', image: 'buzz-cut' },
  { keyword: '2 block', image: 'two-block' },
  { keyword: 'two block', image: 'two-block' },
  { keyword: 'fringe', image: 'fringe' },
  { keyword: 'mullet', image: 'mullet' },
  { keyword: 'crop', image: 'textured-crop' },
  { keyword: 'fade', image: 'fade' },
  { keyword: 'side part', image: 'side-part' },
  { keyword: 'combover', image: 'combover' },
  { keyword: 'comb over', image: 'combover' },
  { keyword: 'beard', image: 'beard' },
];

const FALLBACK_IMAGE = 'textured-crop';

export function getStyleImage(styleName) {
  const name = (styleName || '').toLowerCase();
  const match = STYLE_IMAGE_RULES.find((rule) => name.includes(rule.keyword));
  return `/haircuts/${match ? match.image : FALLBACK_IMAGE}.jpg`;
}