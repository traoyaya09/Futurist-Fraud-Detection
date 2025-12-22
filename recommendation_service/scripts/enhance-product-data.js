/**
 * scripts/enhance-product-data.js
 * 🧠 SMART AI-POWERED PRODUCT DATA ENHANCEMENT
 * 
 * Optimized for large datasets (52K+ products)
 * - Batch processing with validation
 * - Handles edge cases and empty fields
 * - Real-time progress indicators
 * 
 * Usage: node scripts/enhance-product-data.js
 */

const mongoose = require('mongoose');
require('dotenv').config();

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://traoyaya09:vcqgF9ub9r57oq0m@cluster0.lbqbl2z.mongodb.net/futurist_e-commerce';
const BATCH_SIZE = 100;

// Progress bar utility
function createProgressBar(current, total, label = '') {
  const percentage = Math.round((current / total) * 100);
  const filled = Math.round(percentage / 2);
  const empty = 50 - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  
  process.stdout.write(`\r${label} [${bar}] ${percentage}% (${current}/${total})`);
  
  if (current === total) {
    console.log('');
  }
}

// Time formatting utility
function formatTime(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (minutes < 60) return `${minutes}m ${secs}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m ${secs}s`;
}

/**
 * Validate and clean update data
 */
function validateUpdateData(data) {
  const cleaned = {};
  
  // Only include non-empty, valid fields
  Object.keys(data).forEach(key => {
    const value = data[key];
    
    // Skip undefined, null, or empty strings (except for valid empty arrays/objects)
    if (value === undefined || value === null) return;
    if (typeof value === 'string' && value.trim() === '') return;
    
    // Handle arrays
    if (Array.isArray(value)) {
      if (value.length > 0) {
        cleaned[key] = value;
      }
      return;
    }
    
    // Handle objects (but not MongoDB ObjectIds)
    if (typeof value === 'object' && !value._bsontype) {
      const cleanedObj = validateUpdateData(value);
      if (Object.keys(cleanedObj).length > 0) {
        cleaned[key] = cleanedObj;
      }
      return;
    }
    
    // Include all other valid values
    cleaned[key] = value;
  });
  
  return cleaned;
}

/**
 * 🧠 SMART ANALYZER
 */
class SmartProductAnalyzer {
  constructor() {
    this.categoryPatterns = {};
  }

  async analyzeCategory(Product, category) {
    console.log(`\n   🔍 Analyzing category: ${category}`);
    console.log(`   ⏳ Searching for complete products...`);
    
    const completeProducts = await Product.find({
      category: category,
      $and: [
        { description: { $exists: true, $ne: '', $ne: 'No description available' } },
        { features: { $exists: true, $not: { $size: 0 } } },
        { specifications: { $exists: true } }
      ]
    }).limit(50).lean();

    console.log(`   📦 Found ${completeProducts.length} complete products`);

    if (completeProducts.length === 0) {
      console.log(`      No complete products found. Using intelligent defaults.`);
      return this.createDefaultPatterns(category);
    }

    console.log(`   🧮 Extracting patterns...`);

    const patterns = {
      category: category,
      sampleSize: completeProducts.length,
      commonFeatures: this.extractCommonFeatures(completeProducts),
      commonSpecs: this.extractCommonSpecs(completeProducts),
      commonTags: this.extractCommonTags(completeProducts),
      descriptionPatterns: this.extractDescriptionPatterns(completeProducts),
      averageWeight: this.calculateAverage(completeProducts, 'shipping.weight'),
      averageDimensions: this.calculateAverageDimensions(completeProducts),
      shippingPatterns: this.analyzeShippingPatterns(completeProducts),
      priceRange: this.analyzePriceRange(completeProducts),
      commonBrands: this.extractCommonBrands(completeProducts),
      variantPatterns: this.analyzeVariantPatterns(completeProducts)
    };

    console.log(`     Pattern extraction complete!`);
    this.categoryPatterns[category] = patterns;
    return patterns;
  }

  createDefaultPatterns(category) {
    return {
      category: category,
      sampleSize: 0,
      commonFeatures: [
        'High quality product',
        'Great value for money',
        'Durable and reliable',
        'Easy to use',
        'Trusted brand',
        'Customer satisfaction guaranteed'
      ],
      commonSpecs: {
        'Category': category,
        'Quality': 'Premium',
        'Warranty': '6 Months'
      },
      commonTags: [category.toLowerCase().replace(/[^a-z0-9]/g, '-'), 'quality', 'value'],
      descriptionPatterns: {
        commonWords: ['quality', 'premium', 'durable', 'reliable'],
        commonPhrases: []
      },
      averageWeight: 0.5,
      averageDimensions: { length: 20, width: 15, height: 10 },
      shippingPatterns: {
        freeShippingPercentage: 0,
        averageCost: 40,
        mostCommonDelivery: '3-5 business days'
      },
      priceRange: null,
      commonBrands: [],
      variantPatterns: []
    };
  }

  extractCommonFeatures(products) {
    process.stdout.write(`      → Extracting features...`);
    const featureCount = {};
    
    products.forEach(product => {
      if (product.features && Array.isArray(product.features)) {
        product.features.forEach(feature => {
          if (feature && typeof feature === 'string' && feature.trim()) {
            const normalized = this.normalizeText(feature);
            featureCount[normalized] = (featureCount[normalized] || 0) + 1;
          }
        });
      }
    });

    const result = Object.entries(featureCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([feature]) => feature);
    
    console.log(` Found ${result.length}`);
    return result.length > 0 ? result : ['High quality product', 'Great value for money'];
  }

  extractCommonSpecs(products) {
    process.stdout.write(`      → Extracting specifications...`);
    const specKeys = {};
    const specValues = {};

    products.forEach(product => {
      if (product.specifications) {
        const specs = product.specifications instanceof Map 
          ? Object.fromEntries(product.specifications) 
          : product.specifications;

        if (specs && typeof specs === 'object') {
          Object.entries(specs).forEach(([key, value]) => {
            if (key && value) {
              const normalizedKey = this.normalizeText(key);
              specKeys[normalizedKey] = (specKeys[normalizedKey] || 0) + 1;
              
              if (!specValues[normalizedKey]) {
                specValues[normalizedKey] = [];
              }
              specValues[normalizedKey].push(value);
            }
          });
        }
      }
    });

    const commonKeys = Object.entries(specKeys)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([key]) => key);

    const commonSpecs = {};
    commonKeys.forEach(key => {
      const values = specValues[key];
      const valueCounts = {};
      values.forEach(v => {
        valueCounts[v] = (valueCounts[v] || 0) + 1;
      });
      
      const mostCommon = Object.entries(valueCounts)
        .sort((a, b) => b[1] - a[1])[0];
      
      commonSpecs[key] = mostCommon ? mostCommon[0] : 'Standard';
    });

    console.log(` Found ${Object.keys(commonSpecs).length}`);
    return Object.keys(commonSpecs).length > 0 ? commonSpecs : { 'Quality': 'Premium' };
  }

  extractCommonTags(products) {
    process.stdout.write(`      → Extracting tags...`);
    const tagCount = {};
    
    products.forEach(product => {
      if (product.tags && Array.isArray(product.tags)) {
        product.tags.forEach(tag => {
          if (tag && typeof tag === 'string' && tag.trim()) {
            const normalized = tag.toLowerCase().trim();
            tagCount[normalized] = (tagCount[normalized] || 0) + 1;
          }
        });
      }
    });

    const result = Object.entries(tagCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([tag]) => tag);
    
    console.log(` Found ${result.length}`);
    return result.length > 0 ? result : ['quality', 'value'];
  }

  extractDescriptionPatterns(products) {
    process.stdout.write(`      → Analyzing descriptions...`);
    const words = {};

    products.forEach(product => {
      if (product.description && product.description !== 'No description available') {
        const descWords = product.description.toLowerCase()
          .split(/\s+/)
          .filter(word => word.length > 4);
        
        descWords.forEach(word => {
          words[word] = (words[word] || 0) + 1;
        });
      }
    });

    const result = {
      commonWords: Object.entries(words)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([word]) => word),
      commonPhrases: []
    };
    
    console.log(` Found ${result.commonWords.length} words`);
    return result;
  }

  calculateAverage(products, path) {
    const values = products
      .map(p => this.getNestedValue(p, path))
      .filter(v => v && v > 0);
    
    if (values.length === 0) return 0.5;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  calculateAverageDimensions(products) {
    const dimensions = products
      .map(p => p.shipping?.dimensions)
      .filter(d => d && d.length && d.width && d.height);
    
    if (dimensions.length === 0) return { length: 20, width: 15, height: 10 };

    return {
      length: Math.round(dimensions.reduce((a, b) => a + b.length, 0) / dimensions.length),
      width: Math.round(dimensions.reduce((a, b) => a + b.width, 0) / dimensions.length),
      height: Math.round(dimensions.reduce((a, b) => a + b.height, 0) / dimensions.length)
    };
  }

  analyzeShippingPatterns(products) {
    const shippingData = products
      .map(p => p.shipping)
      .filter(s => s);

    if (shippingData.length === 0) {
      return {
        freeShippingPercentage: 0,
        averageCost: 40,
        mostCommonDelivery: '3-5 business days'
      };
    }

    const freeShippingCount = shippingData.filter(s => s.freeShipping).length;
    const deliveryTimes = {};
    const costs = shippingData.map(s => s.shippingCost || 0).filter(c => c > 0);

    shippingData.forEach(s => {
      if (s.estimatedDelivery) {
        deliveryTimes[s.estimatedDelivery] = (deliveryTimes[s.estimatedDelivery] || 0) + 1;
      }
    });

    const mostCommonDelivery = Object.entries(deliveryTimes)
      .sort((a, b) => b[1] - a[1])[0];

    return {
      freeShippingPercentage: (freeShippingCount / shippingData.length) * 100,
      averageCost: costs.length > 0 ? costs.reduce((a, b) => a + b, 0) / costs.length : 40,
      mostCommonDelivery: mostCommonDelivery ? mostCommonDelivery[0] : '3-5 business days'
    };
  }

  analyzePriceRange(products) {
    const prices = products.map(p => p.price).filter(p => p > 0);
    
    if (prices.length === 0) return null;

    return {
      min: Math.min(...prices),
      max: Math.max(...prices),
      average: prices.reduce((a, b) => a + b, 0) / prices.length,
      median: this.calculateMedian(prices)
    };
  }

  extractCommonBrands(products) {
    const brands = {};
    
    products.forEach(product => {
      if (product.brand && product.brand.trim() !== '') {
        brands[product.brand] = (brands[product.brand] || 0) + 1;
      }
    });

    return Object.entries(brands)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([brand]) => brand);
  }

  analyzeVariantPatterns(products) {
    const variantTypes = {};
    
    products.forEach(product => {
      if (product.variants && product.variants.length > 0) {
        product.variants.forEach(variant => {
          if (variant.name) {
            variantTypes[variant.name] = (variantTypes[variant.name] || 0) + 1;
          }
        });
      }
    });

    return Object.entries(variantTypes)
      .sort((a, b) => b[1] - a[1])
      .map(([type]) => type);
  }

  enhanceProductSmart(product, patterns) {
    if (!product || !product.name) {
      throw new Error('Invalid product data');
    }

    const enhanced = {};

    // Only update fields that need enhancement
    if (!product.description || product.description === 'No description available' || product.description.trim() === '') {
      enhanced.description = this.generateSmartDescription(product, patterns);
    }

    if (!product.shortDescription || product.shortDescription.trim() === '') {
      enhanced.shortDescription = this.generateSmartShortDescription(product, patterns);
    }

    if (!product.features || !Array.isArray(product.features) || product.features.length === 0) {
      enhanced.features = this.generateSmartFeatures(product, patterns);
    }

    if (!product.specifications || Object.keys(product.specifications || {}).length === 0) {
      enhanced.specifications = this.generateSmartSpecifications(product, patterns);
    }

    if (!product.tags || !Array.isArray(product.tags) || product.tags.length === 0) {
      enhanced.tags = this.generateSmartTags(product, patterns);
    }

    if (!product.shipping || !product.shipping.weight || product.shipping.weight === 0) {
      enhanced.shipping = this.generateSmartShipping(product, patterns);
    }

    if (!product.brand || product.brand.trim() === '') {
      enhanced.brand = this.extractBrandFromName(product.name, patterns.commonBrands);
    }

    if (!product.manufacturer || product.manufacturer.trim() === '') {
      enhanced.manufacturer = product.brand || enhanced.brand || patterns.commonBrands[0] || 'Quality Manufacturer';
    }

    if (!product.modelNumber || product.modelNumber.trim() === '') {
      const modelNum = this.extractModelNumber(product.name);
      if (modelNum) {
        enhanced.modelNumber = modelNum;
      }
    }

    // Inventory enhancements
    if (product.inventory) {
      if (!product.inventory.warehouse) {
        if (!enhanced.inventory) enhanced.inventory = { ...product.inventory };
        enhanced.inventory.warehouse = this.selectWarehouse(product);
      }
      if (!product.inventory.supplier) {
        if (!enhanced.inventory) enhanced.inventory = { ...product.inventory };
        enhanced.inventory.supplier = enhanced.manufacturer || product.manufacturer || 'Quality Supplier';
      }
    }

    // Badges
    if (product.rating >= 4.5 && product.reviewsCount >= 100 && !product.isFeatured) {
      enhanced.isFeatured = true;
    }
    if ((product.reviewsCount >= 300 || product.rating >= 4.6) && !product.isBestseller) {
      enhanced.isBestseller = true;
    }

    // SEO enhancements
    if (product.seo) {
      if (!product.seo.keywords || product.seo.keywords.length === 0) {
        if (!enhanced.seo) enhanced.seo = { ...product.seo };
        enhanced.seo.keywords = [...new Set([
          enhanced.brand || product.brand,
          product.category,
          product.subCategory,
          ...(enhanced.tags || product.tags || []).slice(0, 5)
        ].filter(Boolean))];
      }
      if (product.seo.metaDescription === 'No description available') {
        if (!enhanced.seo) enhanced.seo = { ...product.seo };
        enhanced.seo.metaDescription = enhanced.shortDescription || product.shortDescription || product.name;
      }
    }

    return enhanced;
  }

  generateSmartDescription(product, patterns) {
    const name = product.name;
    const nameWords = name.toLowerCase().split(/\s+/);
    const keyFeatures = nameWords.filter(word => word.length > 4).slice(0, 3).join(', ');

    let description = `${name} is a premium quality product designed to meet your needs. `;
    description += `Features include ${keyFeatures}, ensuring excellent performance and value. `;
    description += `Perfect for daily use, this product combines quality, durability, and style. `;
    
    if (patterns.priceRange && product.price < patterns.priceRange.average * 0.7) {
      description += `Offering exceptional value at an affordable price point. `;
    } else if (patterns.priceRange && product.price > patterns.priceRange.average * 1.3) {
      description += `A premium choice for those seeking the highest quality. `;
    }

    description += `Trusted by thousands of satisfied customers. Order now for fast delivery!`;
    return description;
  }

  generateSmartShortDescription(product, patterns) {
    const nameWords = product.name.split(' ').slice(0, 8).join(' ');
    if (patterns.commonFeatures && patterns.commonFeatures.length > 0) {
      return `${nameWords} - ${patterns.commonFeatures[0]}`;
    }
    return `Premium ${nameWords} - Great quality and value`;
  }

  generateSmartFeatures(product, patterns) {
    const features = [];
    const nameWords = product.name.toLowerCase();

    if (patterns.commonFeatures && patterns.commonFeatures.length > 0) {
      features.push(...patterns.commonFeatures.slice(0, 4));
    }

    if (nameWords.includes('premium') || nameWords.includes('luxury')) {
      features.push('Premium quality construction');
    }
    if (product.rating >= 4.5) {
      features.push('Highly rated by customers');
    }

    while (features.length < 6) {
      features.push('Excellent value for money');
      features.push('Trusted brand quality');
      features.push('Easy to use and maintain');
    }

    return [...new Set(features)].slice(0, 6);
  }

  generateSmartSpecifications(product, patterns) {
    const specs = { ...patterns.commonSpecs };
    if (product.brand) specs['Brand'] = product.brand;
    specs['Category'] = product.category;
    if (product.subCategory) specs['Subcategory'] = product.subCategory;

    if (product.price > 1000) {
      specs['Warranty'] = '1 Year Manufacturer Warranty';
    } else if (product.price > 500) {
      specs['Warranty'] = '6 Months Warranty';
    } else {
      specs['Warranty'] = '3 Months Warranty';
    }

    return specs;
  }

  generateSmartTags(product, patterns) {
    const tags = new Set();
    
    if (patterns.commonTags && patterns.commonTags.length > 0) {
      patterns.commonTags.slice(0, 4).forEach(tag => tags.add(tag));
    }
    
    if (product.category) tags.add(product.category.toLowerCase().replace(/[^a-z0-9]/g, '-'));
    if (product.subCategory) tags.add(product.subCategory.toLowerCase().replace(/[^a-z0-9]/g, '-'));
    if (product.brand) tags.add(product.brand.toLowerCase().replace(/[^a-z0-9]/g, '-'));
    
    return Array.from(tags).slice(0, 8);
  }

  generateSmartShipping(product, patterns) {
    return {
      weight: patterns.averageWeight || 0.5,
      dimensions: patterns.averageDimensions || { length: 20, width: 15, height: 10 },
      freeShipping: product.price > 1000 || (patterns.shippingPatterns && patterns.shippingPatterns.freeShippingPercentage > 50),
      shippingCost: product.price > 1000 ? 0 : Math.round(patterns.shippingPatterns?.averageCost || 40),
      estimatedDelivery: patterns.shippingPatterns?.mostCommonDelivery || '3-5 business days'
    };
  }

  extractBrandFromName(name, commonBrands) {
    const firstWord = name.split(' ')[0];
    if (commonBrands && commonBrands.length > 0) {
      return commonBrands.find(brand => brand.toLowerCase() === firstWord.toLowerCase()) || firstWord;
    }
    return firstWord;
  }

  extractModelNumber(name) {
    const modelMatch = name.match(/\b([A-Z]{2,}[-]?[0-9]{2,}[A-Z0-9]*)\b/i);
    return modelMatch ? modelMatch[1] : '';
  }

  selectWarehouse(product) {
    const warehouses = [
      'Central Warehouse - Mumbai',
      'North Distribution Center - Delhi',
      'South Regional Hub - Bangalore',
      'East Coast Facility - Kolkata',
      'West Zone Warehouse - Pune'
    ];
    const hash = product.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return warehouses[hash % warehouses.length];
  }

  normalizeText(text) {
    return text.toString().toLowerCase().trim();
  }

  getNestedValue(obj, path) {
    return path.split('.').reduce((current, prop) => current?.[prop], obj);
  }

  calculateMedian(numbers) {
    const sorted = numbers.sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
  }
}

/**
 * Main execution with BATCH PROCESSING
 */
async function main() {
  const startTime = Date.now();
  
  try {
    console.log('  SMART PRODUCT ENHANCEMENT SCRIPT (52K OPTIMIZED)');
    console.log('═'.repeat(60));
    console.log('');
    
    console.log('🔌 Connecting to MongoDB...');
    await mongoose.connect(MONGODB_URI, {
      serverSelectionTimeoutMS: 10000,
      socketTimeoutMS: 60000,
    });
    console.log('  Connected to futurist_e-commerce database');
    console.log('');
    
    const Product = mongoose.model('Product', new mongoose.Schema({}, { strict: false, collection: 'products' }));
    
    console.log('📊 Counting products to enhance...');
    const totalCount = await Product.countDocuments({
      $or: [
        { description: 'No description available' },
        { description: { $exists: false } },
        { description: '' },
        { shortDescription: '' },
        { shortDescription: { $exists: false } },
        { features: { $size: 0 } },
        { features: { $exists: false } },
        { tags: { $size: 0 } },
        { tags: { $exists: false } },
        { manufacturer: '' },
        { manufacturer: { $exists: false } }
      ]
    });
    
    console.log(`  Found ${totalCount.toLocaleString()} products to enhance`);
    console.log(`📦 Processing in batches of ${BATCH_SIZE}`);
    console.log('');
    
    if (totalCount === 0) {
      console.log('✨ All products are already enhanced!');
      return;
    }

    const analyzer = new SmartProductAnalyzer();

    console.log('📂 Fetching categories...');
    const categories = await Product.distinct('category');
    console.log(`  Found ${categories.length} categories`);
    console.log('');

    let totalEnhanced = 0;
    let totalFailed = 0;

    for (let catIndex = 0; catIndex < categories.length; catIndex++) {
      const category = categories[catIndex] || 'uncategorized';
      
      console.log('═'.repeat(60));
      console.log(`📁 Category ${catIndex + 1}/${categories.length}: ${category}`);
      console.log('─'.repeat(60));

      const patterns = await analyzer.analyzeCategory(Product, category);

      if (patterns && patterns.sampleSize > 0) {
        console.log('');
        console.log('   📊 Pattern Analysis Results:');
        console.log(`   ├─ Sample size: ${patterns.sampleSize} products`);
        console.log(`   ├─ Common features: ${patterns.commonFeatures.length}`);
        console.log(`   ├─ Common specs: ${Object.keys(patterns.commonSpecs).length}`);
        console.log(`   ├─ Common tags: ${patterns.commonTags.length}`);
        if (patterns.priceRange) {
          console.log(`   └─ Price range: $${patterns.priceRange.min.toFixed(0)} - $${patterns.priceRange.max.toFixed(0)}`);
        }
        console.log('');
      }

      const categoryCount = await Product.countDocuments({
        category: category,
        $or: [
          { description: 'No description available' },
          { description: { $exists: false } },
          { description: '' },
          { features: { $size: 0 } },
          { features: { $exists: false } }
        ]
      });

      console.log(`   🔧 Enhancing ${categoryCount} products in batches...`);
      console.log('');

      let categoryEnhanced = 0;
      let skip = 0;

      while (skip < categoryCount) {
        const batch = await Product.find({
          category: category,
          $or: [
            { description: 'No description available' },
            { description: { $exists: false } },
            { description: '' },
            { features: { $size: 0 } },
            { features: { $exists: false } }
          ]
        }).skip(skip).limit(BATCH_SIZE).lean();

        if (batch.length === 0) break;

        const bulkOps = [];
        
        for (const product of batch) {
          try {
            const enhancedData = analyzer.enhanceProductSmart(product, patterns);
            
            // Validate and clean the update data
            const cleanedData = validateUpdateData(enhancedData);
            
            // Only add to bulk ops if there's data to update
            if (Object.keys(cleanedData).length > 0) {
              bulkOps.push({
                updateOne: {
                  filter: { _id: product._id },
                  update: { $set: cleanedData }
                }
              });
              categoryEnhanced++;
            }
          } catch (err) {
            totalFailed++;
            console.log(`\n      Skipped product: ${err.message}`);
          }
        }

        if (bulkOps.length > 0) {
          try {
            await Product.bulkWrite(bulkOps, { ordered: false });
          } catch (bulkErr) {
            console.log(`\n      Bulk write error: ${bulkErr.message}`);
            totalFailed += bulkOps.length;
          }
        }

        skip += BATCH_SIZE;
        const progress = Math.min(skip, categoryCount);
        createProgressBar(progress, categoryCount, '   Progress');
      }

      totalEnhanced += categoryEnhanced;
      console.log(`     Category complete! (${categoryEnhanced} products enhanced)`);
      console.log('');

      const overallProgress = Math.round((totalEnhanced / totalCount) * 100);
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = totalEnhanced / elapsed;
      const remaining = rate > 0 ? (totalCount - totalEnhanced) / rate : 0;
      
      console.log(`   📊 Overall: ${totalEnhanced.toLocaleString()}/${totalCount.toLocaleString()} (${overallProgress}%)`);
      console.log(`   ⏱️  Elapsed: ${formatTime(elapsed)} | ETA: ${formatTime(remaining)}`);
      console.log('');
    }
    
    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;
    
    console.log('═'.repeat(60));
    console.log('🎉 ENHANCEMENT COMPLETE!');
    console.log('═'.repeat(60));
    console.log(`  Successfully enhanced: ${totalEnhanced.toLocaleString()}`);
    console.log(`  Failed/Skipped: ${totalFailed.toLocaleString()}`);
    console.log(`📦 Total processed: ${(totalEnhanced + totalFailed).toLocaleString()}`);
    console.log(`📂 Categories processed: ${categories.length}`);
    console.log(`⏱️  Time taken: ${formatTime(duration)}`);
    if (duration > 0) {
      console.log(`⚡ Speed: ${(totalEnhanced / duration).toFixed(2)} products/sec`);
    }
    console.log('═'.repeat(60));
    console.log('');
    
  } catch (error) {
    console.error('');
    console.error('  ERROR:', error.message);
    console.error('Stack:', error.stack);
    throw error;
  } finally {
    await mongoose.disconnect();
    console.log('🔌 Disconnected from MongoDB');
  }
}

if (require.main === module) {
  main()
    .then(() => {
      console.log('✨ Script completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('💥 Script failed:', error.message);
      process.exit(1);
    });
}

module.exports = { SmartProductAnalyzer };
