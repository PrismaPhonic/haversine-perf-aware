use proc_macro::TokenStream;
use quote::quote;
use rdtsc_timer_rogflow::Profiler;
use syn::ItemFn;

/// Time the duration of a function, either to stdout or via `tracing`.
#[proc_macro_attribute]
pub fn time_function(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as ItemFn);

    let func_name = &input.sig.ident;
    let func_block = &input.block;
    let func_output = &input.sig.output;
    let func_input = &input.sig.inputs;
    let func_vis = &input.vis;

    let func_label = if args.is_empty() {
        format!("{func_name}()")
    } else {
        args.to_string()
    };

    let output = quote! {
        #func_vis fn #func_name(#func_input) #func_output {
            let start = ::rdtsc_timer_rogflow::cpu_timer();
            let result = (|| #func_block)();
            let end = ::rdtsc_timer_rogflow::cpu_timer();
            let cycles = end - start;
            #[cfg(not(feature = "tracing"))]
            println!("`{}` took {} cycles", #func_label, cycles);
            #[cfg(feature = "tracing")]
            ::tracing::trace!("`{}` took {} cycles", #func_label, cycles);
            result
        }
    };

    output.into()
}

/// Time the duration of code snippet, either to stdout or via `tracing`.
#[proc_macro]
pub fn time_snippet(input: TokenStream) -> TokenStream {
    let block: proc_macro2::token_stream::TokenStream = input.into();

    let output = quote! {
        {
            let begin = line!();
            let start = ::rdtsc_timer_rogflow::cpu_timer();
            let result =
                #block;
            let end = ::rdtsc_timer_rogflow::cpu_timer();
            let cycles = end - start;
            #[cfg(not(feature = "tracing"))]
            println!("{}:{} took {} cycles", file!(), begin, cycles);
            #[cfg(feature = "tracing")]
            ::tracing::trace!("{}:{} took {:?}.", file!(), begin, cycles);
            result
        }
    };

    output.into()
}
