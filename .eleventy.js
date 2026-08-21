const { katex } = require("@mdit/plugin-katex");
const { DateTime } = require("luxon");

module.exports = function(eleventyConfig) {
  eleventyConfig.addPassthroughCopy("src/.nojekyll");
  eleventyConfig.addPassthroughCopy("src/js");
  eleventyConfig.addPassthroughCopy("src/css");
  eleventyConfig.addPassthroughCopy("src/assets");

  eleventyConfig.addFilter("readableDate", (dateObj) => {
    return new Date(dateObj).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      timeZone: "UTC"
    });
  });

  eleventyConfig.addFilter("numericDate", (dateObj) => {
      return DateTime.fromJSDate(dateObj, { zone: "utc" }).toFormat("MM.dd.yy");
    });

  eleventyConfig.amendLibrary("md", (mdLib) => mdLib.use(katex));

  return {
    pathPrefix: "/",
    dir: {
      input: "src",
      output: "_site"
    }
  };
};
